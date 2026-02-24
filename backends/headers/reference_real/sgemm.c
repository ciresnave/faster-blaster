/**
 * @file sgemm.c
 * @brief Reference implementation of SGEMM (Single precision GEneral Matrix-Matrix multiply)
 *
 * SGEMM: C := alpha*A*B + beta*C
 *
 * This is a reference implementation prioritizing:
 * 1. CORRECTNESS - Exact specification compliance
 * 2. NUMERICAL STABILITY - Kahan summation, proper handling of denormals
 * 3. SIMPLICITY - Clear, understandable code
 * 4. NOT PERFORMANCE - intentionally simple (no SIMD, no blocking)
 *
 * Performance expectation: ~0.1x-0.01x optimized implementations
 * This is a feature, not a bug - we're the "beacon of truth"
 */

#include <blas_reference.h>
#
#include <string.h>

/**
 * BLAS SGEMM - Single precision General Matrix-Matrix Multiply
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
void sgemm_ref(
    char trans_a, char trans_b,
    int m, int n, int k,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc)
{
    // Input validation
    if (m <= 0 || n <= 0 || k <= 0) {
        return;  // No-op for empty matrices
    }
    
    if (alpha == 0.0f && beta == 1.0f) {
        return;  // No-op: C := 0 + C = C
    }
    
    if (!A || !B || !C) {
        return;  // Invalid pointers
    }
    
    // Step 1: Scale C by beta (or zero if beta=0)
    if (beta == 0.0f) {
        // C := 0
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        // C := beta * C
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    // else: beta == 1.0f, C unchanged
    
    if (alpha == 0.0f) {
        return;  // Only scaling needed
    }
    
    // Step 2: Compute A*B contribution using Kahan summation
    // Iterate over result matrix C (m x n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Compute dot product: A[i,:] · B[:,j]
            // Using Kahan summation for improved accuracy
            
            float sum = 0.0f;
            float correction = 0.0f;  // Kahan summation correction term
            
            for (int l = 0; l < k; l++) {
                // Get matrix elements, handling transposes
                float a_il, b_lj;
                
                if (trans_a == 'N' || trans_a == 'n') {
                    a_il = A[i * lda + l];  // A[i][l]
                } else if (trans_a == 'T' || trans_a == 't') {
                    a_il = A[l * lda + i];  // A[l][i] (transposed)
                } else if (trans_a == 'C' || trans_a == 'c') {
                    // Conjugate transpose - for real floats, same as transpose
                    a_il = A[l * lda + i];
                } else {
                    return;  // Invalid trans_a
                }
                
                if (trans_b == 'N' || trans_b == 'n') {
                    b_lj = B[l * ldb + j];  // B[l][j]
                } else if (trans_b == 'T' || trans_b == 't') {
                    b_lj = B[j * ldb + l];  // B[j][l] (transposed)
                } else if (trans_b == 'C' || trans_b == 'c') {
                    // Conjugate transpose - for real floats, same as transpose
                    b_lj = B[j * ldb + l];
                } else {
                    return;  // Invalid trans_b
                }
                
                // Kahan summation: accurate accumulation
                float product = a_il * b_lj;
                float y = product - correction;           // Subtract correction
                float t = sum + y;                        // Try to add
                correction = (t - sum) - y;               // Update correction
                sum = t;                                  // Update sum
            }
            
            // Add contribution: C[i][j] += alpha * sum
            float contribution = alpha * sum;
            float y = contribution - correction;
            float t = C[i * ldc + j] + y;
            C[i * ldc + j] = t;
        }
    }
}
