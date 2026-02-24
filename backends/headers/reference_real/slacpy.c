/* slacpy - copy matrix */
void slacpy_(const char *uplo, const int *m, const int *n, const float *A, const int *lda, 
             float *B, const int *ldb)
{
    int mm = *m;
    int nn = *n;
    int lda_val = *lda;
    int ldb_val = *ldb;
    char uplo_val = *uplo;
    
    if (uplo_val == 'U' || uplo_val == 'u') {
        // Copy upper triangle
        for (int j = 0; j < nn; j++) {
            int i_max = (j + 1 < mm) ? j + 1 : mm;
            for (int i = 0; i < i_max; i++) {
                B[i + j * ldb_val] = A[i + j * lda_val];
            }
        }
    } else if (uplo_val == 'L' || uplo_val == 'l') {
        // Copy lower triangle
        for (int j = 0; j < nn; j++) {
            for (int i = j; i < mm; i++) {
                B[i + j * ldb_val] = A[i + j * lda_val];
            }
        }
    } else {
        // Copy all elements
        for (int j = 0; j < nn; j++) {
            for (int i = 0; i < mm; i++) {
                B[i + j * ldb_val] = A[i + j * lda_val];
            }
        }
    }
}
