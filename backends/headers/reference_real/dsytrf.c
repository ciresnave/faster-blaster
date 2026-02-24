/* dsytrf - symmetric indefinite factorization (Bunch-Kaufman) */
void dsytrf_(const char *uplo, const int *n, double *A, const int *lda, int *ipiv, 
             double *work, const int *lwork, int *info)
{
    int nn = *n;
    int lda_val = *lda;
    int lwork_val = *lwork;
    char uplo_val = *uplo;
    
    *info = 0;
    
    // Simplified Bunch-Kaufman: use partial pivoting with 1x1 pivots for now
    // Full implementation would handle 2x2 pivots for better stability
    
    if (uplo_val == 'U' || uplo_val == 'u') {
        // Upper triangular factorization
        for (int k = nn - 1; k >= 0; k--) {
            // Find pivot
            int pivot = k;
            double pivot_val = fabs(A[k + k * lda_val]);
            
            for (int i = 0; i < k; i++) {
                double val = fabs(A[i + k * lda_val]);
                if (val > pivot_val) {
                    pivot_val = val;
                    pivot = i;
                }
            }
            
            ipiv[k] = pivot + 1;
            
            if (pivot_val == 0.0) {
                *info = k + 1;
                return;
            }
            
            // Swap rows and columns if needed
            if (pivot != k) {
                for (int j = 0; j < k; j++) {
                    double temp = A[pivot + j * lda_val];
                    A[pivot + j * lda_val] = A[k + j * lda_val];
                    A[k + j * lda_val] = temp;
                }
                for (int j = pivot + 1; j < k; j++) {
                    double temp = A[pivot + j * lda_val];
                    A[pivot + j * lda_val] = A[j + k * lda_val];
                    A[j + k * lda_val] = temp;
                }
                double temp = A[pivot + pivot * lda_val];
                A[pivot + pivot * lda_val] = A[k + k * lda_val];
                A[k + k * lda_val] = temp;
            }
            
            // Update rows above
            if (k > 0) {
                double d = A[k + k * lda_val];
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < i; j++) {
                        A[j + i * lda_val] -= A[j + k * lda_val] * A[i + k * lda_val] / d;
                    }
                    A[i + i * lda_val] -= A[i + k * lda_val] * A[i + k * lda_val] / d;
                    A[i + k * lda_val] /= d;
                }
            }
        }
    } else {
        // Lower triangular factorization
        for (int k = 0; k < nn; k++) {
            // Find pivot
            int pivot = k;
            double pivot_val = fabs(A[k + k * lda_val]);
            
            for (int i = k + 1; i < nn; i++) {
                double val = fabs(A[i + k * lda_val]);
                if (val > pivot_val) {
                    pivot_val = val;
                    pivot = i;
                }
            }
            
            ipiv[k] = pivot + 1;
            
            if (pivot_val == 0.0) {
                *info = k + 1;
                return;
            }
            
            // Swap rows and columns if needed
            if (pivot != k) {
                for (int j = 0; j < k; j++) {
                    double temp = A[pivot + j * lda_val];
                    A[pivot + j * lda_val] = A[k + j * lda_val];
                    A[k + j * lda_val] = temp;
                }
                for (int j = k + 1; j < pivot; j++) {
                    double temp = A[pivot + j * lda_val];
                    A[pivot + j * lda_val] = A[j + k * lda_val];
                    A[j + k * lda_val] = temp;
                }
                double temp = A[pivot + pivot * lda_val];
                A[pivot + pivot * lda_val] = A[k + k * lda_val];
                A[k + k * lda_val] = temp;
            }
            
            // Update rows below
            if (k < nn - 1) {
                double d = A[k + k * lda_val];
                for (int i = k + 1; i < nn; i++) {
                    for (int j = k + 1; j < i; j++) {
                        A[i + j * lda_val] -= A[i + k * lda_val] * A[j + k * lda_val] / d;
                    }
                    A[i + i * lda_val] -= A[i + k * lda_val] * A[i + k * lda_val] / d;
                    A[i + k * lda_val] /= d;
                }
            }
        }
    }
}
