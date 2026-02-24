/* dlaset - initialize matrix (double precision) */
void dlaset_(const char *uplo, const int *m, const int *n, const double *alpha, const double *beta,
             double *A, const int *lda)
{
    int mm = *m;
    int nn = *n;
    int lda_val = *lda;
    double alpha_val = *alpha;
    double beta_val = *beta;
    char uplo_val = *uplo;
    
    if (uplo_val == 'U' || uplo_val == 'u') {
        // Set upper triangle to alpha, diagonal to beta
        for (int j = 0; j < nn; j++) {
            for (int i = 0; i <= j && i < mm; i++) {
                A[i + j * lda_val] = (i == j) ? beta_val : alpha_val;
            }
        }
    } else if (uplo_val == 'L' || uplo_val == 'l') {
        // Set lower triangle to alpha, diagonal to beta
        for (int j = 0; j < nn; j++) {
            for (int i = j; i < mm; i++) {
                A[i + j * lda_val] = (i == j) ? beta_val : alpha_val;
            }
        }
    } else {
        // Set all to alpha, diagonal to beta
        for (int j = 0; j < nn; j++) {
            for (int i = 0; i < mm; i++) {
                A[i + j * lda_val] = (i == j) ? beta_val : alpha_val;
            }
        }
    }
}
