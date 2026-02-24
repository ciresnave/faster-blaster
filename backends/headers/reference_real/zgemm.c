/**
 * @file zgemm.c
 * @brief Reference implementation of ZGEMM (Complex double precision GEneral Matrix-Matrix multiply)
 *
 * ZGEMM: C := alpha*A*B + beta*C
 *
 * Complex matrices are stored as interleaved real/imaginary pairs:
 * A[i][j] = A_real[i*lda+j] + i*A_imag[i*lda+j] (stored in structure)
 *
 * This is a reference implementation with highest precision:
 * 1. CORRECTNESS - Exact specification compliance with complex arithmetic
 * 2. NUMERICAL STABILITY - Kahan summation with double precision
 * 3. SIMPLICITY - Clear, understandable code
 * 4. NOT PERFORMANCE - intentionally simple (no SIMD, no blocking)
 *
 * Double precision provides ~15-17 decimal digits of accuracy for each component.
 */

#include <blas_reference.h>
#
#include <string.h>

/**
 * Complex double precision number: real + i*imag
 */
typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Multiply two complex numbers
 * (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
 */
static inline complex_d cmul(complex_d a, complex_d b) {
    complex_d result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

/**
 * Conjugate of a complex number
 * conj(a + ib) = a - ib
 */
static inline complex_d conj(complex_d a) {
    complex_d result = {a.real, -a.imag};
    return result;
}

/**
 * Add two complex numbers with Kahan summation
 */
static inline complex_d kahan_add(complex_d sum, complex_d val, complex_d *corr) {
    // Real part
    double y_r = val.real - corr->real;
    double t_r = sum.real + y_r;
    corr->real = (t_r - sum.real) - y_r;
    sum.real = t_r;
    
    // Imaginary part
    double y_i = val.imag - corr->imag;
    double t_i = sum.imag + y_i;
    corr->imag = (t_i - sum.imag) - y_i;
    sum.imag = t_i;
    
    return sum;
}

/**
 * BLAS ZGEMM - Complex double precision General Matrix-Matrix Multiply
 *
 * Computes: C := alpha*A*B + beta*C
 *
 * Matrices are stored in column-major order (Fortran convention)
 * with complex numbers stored as structures containing real and imaginary parts.
 *
 * @param trans_a 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose)
 * @param trans_b 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose)
 * @param m Number of rows of matrices A and C
 * @param n Number of columns of matrices B and C
 * @param k Number of columns of A and rows of B
 * @param alpha Scaling factor for A*B (complex)
 * @param A Matrix A (lda × k if trans_a='N', else lda × m)
 * @param lda Leading dimension of A
 * @param B Matrix B (ldb × n if trans_b='N', else ldb × k)
 * @param ldb Leading dimension of B
 * @param beta Scaling factor for C (complex)
 * @param C Matrix C (ldc × n), UPDATED IN PLACE
 * @param ldc Leading dimension of C
 */
void zgemm_ref(
    char trans_a, char trans_b,
    int m, int n, int k,
    complex_d alpha,
    const complex_d *A, int lda,
    const complex_d *B, int ldb,
    complex_d beta,
    complex_d *C, int ldc)
{
    // Input validation
    if (m <= 0 || n <= 0 || k <= 0) {
        return;  // No-op for empty matrices
    }
    
    if (alpha.real == 0.0 && alpha.imag == 0.0 && beta.real == 1.0 && beta.imag == 0.0) {
        return;  // No-op: C := 0 + C = C
    }
    
    if (!A || !B || !C) {
        return;  // Invalid pointers
    }
    
    // Step 1: Scale C by beta
    if (beta.real == 0.0 && beta.imag == 0.0) {
        // C := 0
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j].real = 0.0;
                C[i * ldc + j].imag = 0.0;
            }
        }
    } else if (!(beta.real == 1.0 && beta.imag == 0.0)) {
        // C := beta * C
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] = cmul(beta, C[i * ldc + j]);
            }
        }
    }
    // else: beta == 1, C unchanged
    
    if (alpha.real == 0.0 && alpha.imag == 0.0) {
        return;  // Only scaling needed
    }
    
    // Step 2: Compute A*B contribution with maximum precision
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Compute dot product: A[i,:] · B[:,j]
            complex_d sum = {0.0, 0.0};
            complex_d correction = {0.0, 0.0};
            
            for (int l = 0; l < k; l++) {
                complex_d a_il, b_lj;
                
                // Get A[i,l] (with transpose handling)
                if (trans_a == 'N' || trans_a == 'n') {
                    a_il = A[i * lda + l];
                } else if (trans_a == 'T' || trans_a == 't') {
                    a_il = A[l * lda + i];
                } else if (trans_a == 'C' || trans_a == 'c') {
                    a_il = conj(A[l * lda + i]);
                } else {
                    return;  // Invalid trans_a
                }
                
                // Get B[l,j] (with transpose handling)
                if (trans_b == 'N' || trans_b == 'n') {
                    b_lj = B[l * ldb + j];
                } else if (trans_b == 'T' || trans_b == 't') {
                    b_lj = B[j * ldb + l];
                } else if (trans_b == 'C' || trans_b == 'c') {
                    b_lj = conj(B[j * ldb + l]);
                } else {
                    return;  // Invalid trans_b
                }
                
                // Multiply and accumulate with Kahan summation
                complex_d product = cmul(a_il, b_lj);
                sum = kahan_add(sum, product, &correction);
            }
            
            // Add contribution: C[i,j] += alpha * sum
            complex_d contribution = cmul(alpha, sum);
            C[i * ldc + j].real += contribution.real;
            C[i * ldc + j].imag += contribution.imag;
        }
    }
}
