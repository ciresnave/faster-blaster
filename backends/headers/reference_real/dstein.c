/* dstein */

#

/* DSTEIN - Compute selected eigenvectors of tridiagonal matrix (double precision)
   N: Matrix size
   D: Diagonal elements
   E: Subdiagonal elements
   M: Number of eigenvalues
   W: Eigenvalues (input)
   IBLOCK: Block index for each eigenvalue
   ISPLIT: Block boundaries
   Z: Output eigenvectors
   LDZ: Leading dimension
   WORK: Workspace
   IWORK: Integer workspace
   IFAIL: Failure information
   INFO: Status
*/
void dstein_(const int *n, const double *d, const double *e, const int *m,
             double *w, const int *iblock, const int *isplit,
             double *z, const int *ldz, double *work, int *iwork, int *ifail,
             int *info)
{
    *info = 0;
    int n_val = *n;
    int m_val = *m;
    int ldz_val = *ldz;

    if (n_val < 0) {
        *info = -1;
        return;
    }
    if (m_val < 0 || m_val > n_val) {
        *info = -4;
        return;
    }
    if (ldz_val < n_val) {
        *info = -8;
        return;
    }

    if (m_val == 0) return;

    /* Initialize eigenvectors */
    for (int j = 0; j < m_val; j++) {
        for (int i = 0; i < n_val; i++) {
            z[i + j * ldz_val] = 0.0;
        }
        z[0 + j * ldz_val] = 1.0;  /* Initial guess */
    }

    /* Inverse iteration for eigenvector computation */
    const int MAX_ITER = 20;
    const double TOL = 1e-12;

    for (int j = 0; j < m_val; j++) {
        double lambda = w[j];
        
        /* Inverse power iteration: (T - lambda*I) * v = u */
        for (int iter = 0; iter < MAX_ITER; iter++) {
            /* Forward substitution on lower part */
            double *v = z + j * ldz_val;
            
            /* Apply (T - lambda*I)^{-1} to v */
            /* Solve (T - lambda*I) y = v */
            double y[512];  /* Workspace for intermediate vector */
            
            for (int i = 0; i < n_val; i++) {
                y[i] = v[i];
            }

            /* Forward elimination */
            for (int i = 0; i < n_val - 1; i++) {
                double diag = d[i] - lambda;
                if (fabs(diag) < 1e-15) {
                    diag = (diag >= 0.0) ? 1e-15 : -1e-15;
                }
                double mult = e[i] / diag;
                y[i + 1] -= mult * y[i];
            }

            /* Back substitution */
            for (int i = n_val - 1; i >= 0; i--) {
                double diag = d[i] - lambda;
                if (fabs(diag) < 1e-15) {
                    diag = (diag >= 0.0) ? 1e-15 : -1e-15;
                }
                y[i] /= diag;
                if (i > 0) {
                    y[i - 1] -= e[i - 1] * y[i];
                }
            }

            /* Normalize */
            double norm = 0.0;
            for (int i = 0; i < n_val; i++) {
                norm += y[i] * y[i];
            }
            norm = sqrt(norm);

            if (norm > 0.0) {
                for (int i = 0; i < n_val; i++) {
                    v[i] = y[i] / norm;
                }
            }
        }

        ifail[j] = 0;  /* Success */
    }
}
