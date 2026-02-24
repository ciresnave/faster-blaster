/**
 * @file dgemm.c
 * @brief Reference implementation of DGEMM (Double precision GEneral Matrix-Matrix multiply)
 *
 * DGEMM: C := alpha*A*B + beta*C
 *
 * This is a reference implementation prioritizing:
 * 1. CORRECTNESS - Exact specification compliance
 * 2. NUMERICAL STABILITY - Kahan summation with double precision
 * 3. SIMPLICITY - Clear, understandable code
 * 4. NOT PERFORMANCE - intentionally simple (no SIMD, no blocking)
 *
 * Double precision provides ~15-17 decimal digits of accuracy
 * compared to ~7 digits for single precision.
 */

#include <blas_reference.h>
#
#include <string.h>

/**
 * BLAS DGEMM - Double precision General Matrix-Matrix Multiply
 *
 * Computes: C := alpha*A*B + beta*C
 *
 * @param trans_a 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose)
 * @param trans_b 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose)
 * @param m Number of rows of matrices A and C
 * @param n Number of columns of matrices B and C
 * @param k Number of columns of A and rows of B
 * @param alpha Scaling factor for A*B
 * @param A Matrix A (lda × k if trans_a='N', else lda × m)
 * @param lda Leading dimension of A (>= max(1, m or k))
 * @param B Matrix B (ldb × n if trans_b='N', else ldb × k)
 * @param ldb Leading dimension of B (>= max(1, k or n))
 * @param beta Scaling factor for C
 * @param C Matrix C (ldc × n), UPDATED IN PLACE
 * @param ldc Leading dimension of C (>= max(1, m))
 */
void dgemm_ref(
    char trans_a, char trans_b,
    int m, int n, int k,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double beta,
    double *C, int ldc)
{
    // Input validation
    if (m <= 0 || n <= 0 || k <= 0) {
        return;  // No-op for empty matrices
    }
    
    if (alpha == 0.0 && beta == 1.0) {
        return;  // No-op: C := 0 + C = C
    }
    
    if (!A || !B || !C) {
        return;  // Invalid pointers
    }
    
    // Step 1: Scale C by beta (or zero if beta=0)
    if (beta == 0.0) {
        // C := 0
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        // C := beta * C
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    // else: beta == 1.0, C unchanged
    
    if (alpha == 0.0) {
        return;  // Only scaling needed
    }
    
    // Step 2: Compute A*B contribution using Kahan summation
    // Iterate over result matrix C (m x n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Compute dot product: A[i,:] · B[:,j]
            // Using Kahan summation for improved accuracy
            
            double sum = 0.0;
            double correction = 0.0;  // Kahan summation correction term
            
            for (int l = 0; l < k; l++) {
                // Get matrix elements, handling transposes
                double a_il, b_lj;
                
                if (trans_a == 'N' || trans_a == 'n') {
                    a_il = A[i * lda + l];  // A[i][l]
                } else if (trans_a == 'T' || trans_a == 't') {
                    a_il = A[l * lda + i];  // A[l][i] (transposed)
                } else if (trans_a == 'C' || trans_a == 'c') {
                    // Conjugate transpose - for real doubles, same as transpose
                    a_il = A[l * lda + i];
                } else {
                    return;  // Invalid trans_a
                }
                
                if (trans_b == 'N' || trans_b == 'n') {
                    b_lj = B[l * ldb + j];  // B[l][j]
                } else if (trans_b == 'T' || trans_b == 't') {
                    b_lj = B[j * ldb + l];  // B[j][l] (transposed)
                } else if (trans_b == 'C' || trans_b == 'c') {
                    // Conjugate transpose - for real doubles, same as transpose
                    b_lj = B[j * ldb + l];
                } else {
                    return;  // Invalid trans_b
                }
                
                // Kahan summation: accurate accumulation
                double product = a_il * b_lj;
                double y = product - correction;         // Subtract correction
                double t = sum + y;                      // Try to add
                correction = (t - sum) - y;              // Update correction
                sum = t;                                 // Update sum
            }
            
            // Add contribution: C[i][j] += alpha * sum
            double contribution = alpha * sum;
            double y = contribution - correction;
            double t = C[i * ldc + j] + y;
            C[i * ldc + j] = t;
        }
    }
}
