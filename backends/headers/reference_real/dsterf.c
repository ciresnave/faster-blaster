/* QR no eigenvectors */

#

/* DSTERF - Compute all eigenvalues of tridiagonal matrix via QR iteration (double precision)
   N: Matrix size
   D: Diagonal elements, overwritten with eigenvalues
   E: Subdiagonal elements
   INFO: Status (0=success, >0=number of non-converged values)
*/
void dsterf_(const int *n, double *d, double *e, int *info)
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
    const double TRESH = 2.22e-16 * 10.0;  /* DBL_EPSILON * 10 */

    int m = n_val - 1;
    int nconv = 0;
    int iter;

    /* Deflate converged eigenvalues */
    for (iter = 0; iter < MAX_ITER; iter++) {
        /* Check for convergence */
        for (int i = 0; i < m; i++) {
            if (fabs(e[i]) < TRESH * (fabs(d[i]) + fabs(d[i + 1]))) {
                e[i] = 0.0;
                nconv++;
            }
        }

        if (nconv == m) break;

        m = 0;
        for (int i = 0; i < n_val - 1; i++) {
            if (e[i] != 0.0) {
                m = i;
            }
        }

        /* QR step via Givens rotations */
        if (m == 0) break;

        /* Compute shift */
        double g = d[m];
        double p = (d[m + 1] - d[m]) / 2.0;
        double r = sqrt(p * p + e[m] * e[m]);
        if (p < 0.0) r = -r;

        d[m] = e[m] + p + r;
        if (r != 0.0) {
            d[m + 1] = e[m] * e[m] / (p + r);
        }

        /* Apply Givens rotations */
        for (int i = m + 1; i < n_val - 1; i++) {
            double f = e[i];
            double c = d[i + 1];
            double b = e[i - 1];

            double a = d[i];
            double h = d[i] - g;
            
            p = (d[i + 1] - d[i]) / 2.0;
            r = sqrt(p * p + e[i] * e[i]);
            if (p < 0.0) r = -r;

            double cs = (p + r) / h;
            double sn = e[i] / h;

            e[i - 1] = f * cs + b * sn;
            d[i] = a * cs * cs + c * sn * sn - 2.0 * b * sn * cs;
            d[i + 1] = a * sn * sn + c * cs * cs + 2.0 * b * sn * cs;
            e[i] = (a - c) * sn * cs + b * (cs * cs - sn * sn);
        }

        g = d[n_val - 1] - g;
        p = (d[n_val - 1] - d[n_val - 2]) / 2.0;
        r = sqrt(p * p + e[n_val - 2] * e[n_val - 2]);
        if (p < 0.0) r = -r;
        
        if (r != 0.0) {
            d[n_val - 1] = e[n_val - 2] * e[n_val - 2] / (p + r) + g;
            e[n_val - 2] = sqrt((e[n_val - 2] * (p + r)) * (e[n_val - 2] / (p + r)));
        }

        nconv = 0;
    }

    /* Count non-converged eigenvalues */
    for (int i = 0; i < n_val - 1; i++) {
        if (e[i] != 0.0) {
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
            double tmp = d[i];
            d[i] = d[k];
            d[k] = tmp;
        }
    }
}

