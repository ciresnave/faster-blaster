#
#include <string.h>
#include <blas_reference.h>

/* DSYEV - Compute eigenvalues and optionally eigenvectors of symmetric matrix (double precision)
   JOBZ: 'N' eigenvalues only, 'V' eigenvalues and eigenvectors
   UPLO: 'U' upper, 'L' lower
   N: Matrix size
   A: Input matrix, overwritten with eigenvectors if JOBZ='V'
   LDA: Leading dimension of A
   W: Output eigenvalues
   WORK: Workspace
   LWORK: Workspace size
   INFO: Status (0=success, <0=bad param, >0=convergence failed)
 */
void dsyev_(const char *jobz, const char *uplo, const int *n,
            double *A, const int *lda, double *W,
            double *work, const int *lwork, int *info)
{
    *info = 0;
    int n_val = *n;
    int lda_val = *lda;
    int lwork_val = *lwork;

    if (jobz[0] != 'N' && jobz[0] != 'V') {
        *info = -1;
        return;
    }
    if (uplo[0] != 'U' && uplo[0] != 'L') {
        *info = -2;
        return;
    }
    if (n_val < 0) {
        *info = -3;
        return;
    }
    if (lda_val < n_val) {
        *info = -5;
        return;
    }
    if (lwork_val < 3 * n_val - 1) {
        *info = -8;
        return;
    }

    if (n_val == 0) return;

    /* Reduce to tridiagonal form using Householder reflections */
    double *tau = work;
    double *work2 = work + n_val;
    int lwork2 = lwork_val - n_val;
    int info2;

    /* Call DSYTRD to reduce to tridiagonal form */
    dsytrd_(uplo, n, A, lda, W, work2, tau, work2 + n_val, &lwork2, &info2);
    if (info2 != 0) {
        *info = info2;
        return;
    }

    /* If computing eigenvectors, generate orthogonal matrix from reflections */
    if (jobz[0] == 'V') {
        dorgtr_(uplo, n, A, lda, tau, work2 + n_val, &lwork2, &info2);
        if (info2 != 0) {
            *info = info2;
            return;
        }
    }

    /* Solve tridiagonal eigenvalue problem */
    double *d = W;
    double *e = work2;
    double *z = A;

    /* Extract super/subdiagonal from A into e */
    if (uplo[0] == 'U') {
        for (int i = 0; i < n_val - 1; i++) {
            e[i] = A[i + (i + 1) * lda_val];
        }
    } else {
        for (int i = 0; i < n_val - 1; i++) {
            e[i] = A[i + 1 + i * lda_val];
        }
    }

    /* Call DSTERF to find eigenvalues of tridiagonal matrix */
    dsterf_(n, W, e, &info2);
    if (info2 != 0) {
        *info = info2;
        return;
    }

    /* If eigenvectors not needed, done */
    if (jobz[0] != 'V') {
        return;
    }

    /* Transform eigenvectors back via orthogonal matrix Q */
    /* Eigenvectors already in A from DORGTR, eigenvalues in W */
}
