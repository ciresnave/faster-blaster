/**
 * @file cgemm.c
 * @brief Reference implementation of CGEMM (Complex single precision GEneral Matrix-Matrix multiply)
 *
 * CGEMM: C := alpha*A*B + beta*C
 *
 * Complex matrices are stored as interleaved real/imaginary pairs:
 * A[i][j] = A_real[2*(i*lda+j)] + i*A_imag[2*(i*lda+j)+1]
 *
 * This is a reference implementation prioritizing:
 * 1. CORRECTNESS - Exact specification compliance with complex arithmetic
 * 2. NUMERICAL STABILITY - Kahan summation for both real and imaginary parts
 * 3. SIMPLICITY - Clear, understandable code
 * 4. NOT PERFORMANCE - intentionally simple (no SIMD, no blocking)
 */

#include <blas_reference.h>
#
#include <string.h>

/**
 * Complex single precision number: real + i*imag
 */
typedef struct {
    float real;
    float imag;
} complex_f;

/**
 * Multiply two complex numbers
 * (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
 */
static inline complex_f cmul(complex_f a, complex_f b) {
    complex_f result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

/**
 * Conjugate of a complex number
 * conj(a + ib) = a - ib
 */
static inline complex_f conj(complex_f a) {
    complex_f result = {a.real, -a.imag};
    return result;
}

/**
 * Add two complex numbers with Kahan summation
 */
static inline complex_f kahan_add(complex_f sum, complex_f val, complex_f *corr) {
    // Real part
    float y_r = val.real - corr->real;
    float t_r = sum.real + y_r;
    corr->real = (t_r - sum.real) - y_r;
    sum.real = t_r;
    
    // Imaginary part
    float y_i = val.imag - corr->imag;
    float t_i = sum.imag + y_i;
    corr->imag = (t_i - sum.imag) - y_i;
    sum.imag = t_i;
    
    return sum;
}

/**
 * BLAS CGEMM - Complex single precision General Matrix-Matrix Multiply
 *
 * Computes: C := alpha*A*B + beta*C
 *
 * Matrices are stored in column-major order (Fortran convention)
 * with complex numbers stored as interleaved real/imaginary pairs.
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
void cgemm_ref(
    char trans_a, char trans_b,
    int m, int n, int k,
    complex_f alpha,
    const complex_f *A, int lda,
    const complex_f *B, int ldb,
    complex_f beta,
    complex_f *C, int ldc)
{
    // Input validation
    if (m <= 0 || n <= 0 || k <= 0) {
        return;  // No-op for empty matrices
    }
    
    if (alpha.real == 0.0f && alpha.imag == 0.0f && beta.real == 1.0f && beta.imag == 0.0f) {
        return;  // No-op: C := 0 + C = C
    }
    
    if (!A || !B || !C) {
        return;  // Invalid pointers
    }
    
    // Step 1: Scale C by beta
    if (beta.real == 0.0f && beta.imag == 0.0f) {
        // C := 0
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j].real = 0.0f;
                C[i * ldc + j].imag = 0.0f;
            }
        }
    } else if (!(beta.real == 1.0f && beta.imag == 0.0f)) {
        // C := beta * C
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] = cmul(beta, C[i * ldc + j]);
            }
        }
    }
    // else: beta == 1, C unchanged
    
    if (alpha.real == 0.0f && alpha.imag == 0.0f) {
        return;  // Only scaling needed
    }
    
    // Step 2: Compute A*B contribution
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Compute dot product: A[i,:] · B[:,j]
            complex_f sum = {0.0f, 0.0f};
            complex_f correction = {0.0f, 0.0f};
            
            for (int l = 0; l < k; l++) {
                complex_f a_il, b_lj;
                
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
                complex_f product = cmul(a_il, b_lj);
                sum = kahan_add(sum, product, &correction);
            }
            
            // Add contribution: C[i,j] += alpha * sum
            complex_f contribution = cmul(alpha, sum);
            C[i * ldc + j].real += contribution.real;
            C[i * ldc + j].imag += contribution.imag;
        }
    }
}
