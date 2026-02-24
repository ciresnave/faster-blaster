/* sstein */

#

/* SSTEIN - Compute selected eigenvectors of tridiagonal matrix
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
void sstein_(const int *n, const float *d, const float *e, const int *m,
             float *w, const int *iblock, const int *isplit,
             float *z, const int *ldz, float *work, int *iwork, int *ifail,
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
            z[i + j * ldz_val] = 0.0f;
        }
        z[0 + j * ldz_val] = 1.0f;  /* Initial guess */
    }

    /* Inverse iteration for eigenvector computation */
    const int MAX_ITER = 20;
    const float TOL = 1e-6f;

    for (int j = 0; j < m_val; j++) {
        float lambda = w[j];
        
        /* Inverse power iteration: (T - lambda*I) * v = u */
        for (int iter = 0; iter < MAX_ITER; iter++) {
            /* Forward substitution on lower part */
            float *v = z + j * ldz_val;
            
            /* Apply (T - lambda*I)^{-1} to v */
            /* Solve (T - lambda*I) y = v */
            float y[512];  /* Workspace for intermediate vector */
            
            for (int i = 0; i < n_val; i++) {
                y[i] = v[i];
            }

            /* Forward elimination */
            for (int i = 0; i < n_val - 1; i++) {
                float diag = d[i] - lambda;
                if (fabsf(diag) < 1e-10f) {
                    diag = (diag >= 0.0f) ? 1e-10f : -1e-10f;
                }
                float mult = e[i] / diag;
                y[i + 1] -= mult * y[i];
            }

            /* Back substitution */
            for (int i = n_val - 1; i >= 0; i--) {
                float diag = d[i] - lambda;
                if (fabsf(diag) < 1e-10f) {
                    diag = (diag >= 0.0f) ? 1e-10f : -1e-10f;
                }
                y[i] /= diag;
                if (i > 0) {
                    y[i - 1] -= e[i - 1] * y[i];
                }
            }

            /* Normalize */
            float norm = 0.0f;
            for (int i = 0; i < n_val; i++) {
                norm += y[i] * y[i];
            }
            norm = sqrtf(norm);

            if (norm > 0.0f) {
                for (int i = 0; i < n_val; i++) {
                    v[i] = y[i] / norm;
                }
            }
        }

        ifail[j] = 0;  /* Success */
    }
}
