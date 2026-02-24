#include <string.h>

/* Forward declarations of Fortran-style functions */
void sgetrf_(const int *m, const int *n, float *A, const int *lda,
             int *ipiv, int *info);
void sgetrs_(const char *trans, const int *n, const int *nrhs,
             const float *A, const int *lda, const int *ipiv,
             float *B, const int *ldb, int *info);

void sgesv_(const int *n, const int *nrhs, float *A, const int *lda, int *ipiv,
            float *B, const int *ldb, int *info)
{
    int lwork = -1;
    float temp_work;
    char trans = 'N';
    
    if (*n <= 0 || *nrhs <= 0) {
        *info = 0;
        return;
    }
    
    /* Compute LU factorization */
    sgetrf_(n, n, A, lda, ipiv, info);
    
    if (*info == 0) {
        /* Solve using LU factorization */
        sgetrs_(&trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    }
}
