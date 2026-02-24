/* QR no eigenvectors */

#
#include <float.h>

/* SSTERF - Compute all eigenvalues of tridiagonal matrix via QR iteration
   N: Matrix size
   D: Diagonal elements, overwritten with eigenvalues
   E: Subdiagonal elements
   INFO: Status (0=success, >0=number of non-converged values)
*/
void ssterf_(const int *n, float *d, float *e, int *info)
{
    *info = 0;
    int n_val = *n;
    
    if (n_val < 0) {
        *info = -1;
        return;
    }
    if (n_val == 0) return;
    if (n_val == 1) return;

    /* QR algorithm for tridiagonal eigenvalues */
    const int MAX_ITER = 30;
    const float TRESH = FLT_EPSILON * 10.0f;

    int m = n_val - 1;
    int nconv = 0;
    int iter;

    /* Deflate converged eigenvalues */
    for (iter = 0; iter < MAX_ITER; iter++) {
        /* Check for convergence */
        for (int i = 0; i < m; i++) {
            if (fabsf(e[i]) < TRESH * (fabsf(d[i]) + fabsf(d[i + 1]))) {
                e[i] = 0.0f;
                nconv++;
            }
        }

        if (nconv == m) break;

        m = 0;
        for (int i = 0; i < n_val - 1; i++) {
            if (e[i] != 0.0f) {
                m = i;
            }
        }

        /* QR step via Givens rotations */
        if (m == 0) break;

        /* Compute shift */
        float g = d[m];
        float p = (d[m + 1] - d[m]) / 2.0f;
        float r = sqrtf(p * p + e[m] * e[m]);
        if (p < 0.0f) r = -r;

        d[m] = e[m] + p + r;
        if (r != 0.0f) {
            d[m + 1] = e[m] * e[m] / (p + r);
        }

        /* Apply Givens rotations */
        for (int i = m + 1; i < n_val - 1; i++) {
            float f = e[i];
            float c = d[i + 1];
            float b = e[i - 1];

            float a = d[i];
            float h = d[i] - g;
            
            p = (d[i + 1] - d[i]) / 2.0f;
            r = sqrtf(p * p + e[i] * e[i]);
            if (p < 0.0f) r = -r;

            float cs = (p + r) / h;
            float sn = e[i] / h;

            e[i - 1] = f * cs + b * sn;
            d[i] = a * cs * cs + c * sn * sn - 2.0f * b * sn * cs;
            d[i + 1] = a * sn * sn + c * cs * cs + 2.0f * b * sn * cs;
            e[i] = (a - c) * sn * cs + b * (cs * cs - sn * sn);
        }

        g = d[n_val - 1] - g;
        p = (d[n_val - 1] - d[n_val - 2]) / 2.0f;
        r = sqrtf(p * p + e[n_val - 2] * e[n_val - 2]);
        if (p < 0.0f) r = -r;
        
        if (r != 0.0f) {
            d[n_val - 1] = e[n_val - 2] * e[n_val - 2] / (p + r) + g;
            e[n_val - 2] = sqrtf((e[n_val - 2] * (p + r)) * (e[n_val - 2] / (p + r)));
        }

        nconv = 0;
    }

    /* Count non-converged eigenvalues */
    for (int i = 0; i < n_val - 1; i++) {
        if (e[i] != 0.0f) {
            (*info)++;
        }
    }

    /* Sort eigenvalues */
    for (int i = 0; i < n_val - 1; i++) {
        int k = i;
        for (int j = i + 1; j < n_val; j++) {
            if (d[j] < d[k]) k = j;
        }
        if (k != i) {
            float tmp = d[i];
            d[i] = d[k];
            d[k] = tmp;
        }
    }
}
