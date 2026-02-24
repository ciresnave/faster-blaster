/* Tridiagonal eigenvalues */

#
#include <float.h>

/* SSTEV - Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix
   JOBZ: 'N' eigenvalues only, 'V' eigenvalues and eigenvectors
   N: Matrix size
   D: Diagonal elements, overwritten with eigenvalues
   E: Subdiagonal elements
   Z: Output eigenvectors (if JOBZ='V')
   LDZ: Leading dimension of Z
   WORK: Workspace
   INFO: Status
 */
void sstev_(const char *jobz, const int *n, float *d, float *e,
            float *z, const int *ldz, float *work, int *info)
{
    *info = 0;
    int n_val = *n;
    int ldz_val = *ldz;

    if (jobz[0] != 'N' && jobz[0] != 'V') {
        *info = -1;
        return;
    }
    if (n_val < 0) {
        *info = -2;
        return;
    }
    if (ldz_val < n_val && jobz[0] == 'V') {
        *info = -6;
        return;
    }

    if (n_val == 0) return;
    if (n_val == 1) {
        if (jobz[0] == 'V') {
            z[0] = 1.0f;
        }
        return;
    }

    /* Find eigenvalues using QR iteration on tridiagonal matrix */
    int info2;
    float *work_qr = work;

    /* Call SSTERF for eigenvalue computation */
    ssterf_(n, d, e, &info2);
    if (info2 != 0) {
        *info = info2;
        return;
    }

    /* If eigenvectors not requested, done */
    if (jobz[0] != 'V') {
        return;
    }

    /* Compute eigenvectors via inverse iteration or similar */
    /* For simplified version, create identity matrix then apply transformations */
    for (int i = 0; i < n_val; i++) {
        for (int j = 0; j < n_val; j++) {
            z[i + j * ldz_val] = (i == j) ? 1.0f : 0.0f;
        }
    }

    /* Apply Givens rotations to accumulate transformations */
    /* This is a simplified version - full implementation would use STEIN */
    sstein_(n, d, e, n, d, z, ldz, work, &info2);
}
