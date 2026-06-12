/**
 * @file faster_blaster_ext.h
 * @brief Extended BLAS operations beyond the standard
 * 
 * This header provides extended functionality including:
 * - Batched operations for multiple independent matrices
 * - Strided batched operations for regularly-spaced matrix arrays
 * - Mixed precision operations
 * - Operation fusion for reduced memory traffic
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_EXT_H
#define FASTER_BLASTER_EXT_H

#include "faster_blaster.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Batched Operations
 * ========================================================================= */

/**
 * @brief Batched single-precision GEMM
 * 
 * Performs multiple independent matrix multiplications:
 *   C[i] = alpha * A[i] * B[i] + beta * C[i]
 * for i = 0 to batch_count-1
 * 
 * This is significantly faster than calling fb_sgemm multiple times,
 * especially on GPUs where kernel launch overhead is amortized.
 * 
 * @param Layout Matrix layout (row-major or column-major)
 * @param TransA Transpose operation for A matrices
 * @param TransB Transpose operation for B matrices
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 * @param alpha Scaling factor for product
 * @param A_array Array of pointers to A matrices
 * @param lda Leading dimension of A matrices
 * @param B_array Array of pointers to B matrices
 * @param ldb Leading dimension of B matrices
 * @param beta Scaling factor for C
 * @param C_array Array of pointers to C matrices (output)
 * @param ldc Leading dimension of C matrices
 * @param batch_count Number of matrix multiplications to perform
 */
void fb_sgemm_batched(const FB_LAYOUT Layout,
                      const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const float alpha,
                      const float **A_array, const int lda,
                      const float **B_array, const int ldb,
                      const float beta,
                      float **C_array, const int ldc,
                      const int batch_count);

void fb_dgemm_batched(const FB_LAYOUT Layout,
                      const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const double alpha,
                      const double **A_array, const int lda,
                      const double **B_array, const int ldb,
                      const double beta,
                      double **C_array, const int ldc,
                      const int batch_count);

void fb_cgemm_batched(const FB_LAYOUT Layout,
                      const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const void *alpha,
                      const void **A_array, const int lda,
                      const void **B_array, const int ldb,
                      const void *beta,
                      void **C_array, const int ldc,
                      const int batch_count);

void fb_zgemm_batched(const FB_LAYOUT Layout,
                      const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const void *alpha,
                      const void **A_array, const int lda,
                      const void **B_array, const int ldb,
                      const void *beta,
                      void **C_array, const int ldc,
                      const int batch_count);

/* ============================================================================
 * Strided Batched Operations
 * ========================================================================= */

/**
 * @brief Strided batched single-precision GEMM
 * 
 * Similar to batched GEMM, but matrices are stored in contiguous memory
 * with constant stride between consecutive matrices. This is more efficient
 * when matrices are stored in a regular array.
 * 
 *   C[i*strideC] = alpha * A[i*strideA] * B[i*strideB] + beta * C[i*strideC]
 * for i = 0 to batch_count-1
 * 
 * @param Layout Matrix layout
 * @param TransA Transpose operation for A matrices
 * @param TransB Transpose operation for B matrices
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 * @param alpha Scaling factor for product
 * @param A Pointer to first A matrix
 * @param lda Leading dimension of A matrices
 * @param strideA Number of elements between consecutive A matrices
 * @param B Pointer to first B matrix
 * @param ldb Leading dimension of B matrices
 * @param strideB Number of elements between consecutive B matrices
 * @param beta Scaling factor for C
 * @param C Pointer to first C matrix (output)
 * @param ldc Leading dimension of C matrices
 * @param strideC Number of elements between consecutive C matrices
 * @param batch_count Number of matrix multiplications
 */
void fb_sgemm_strided_batched(const FB_LAYOUT Layout,
                               const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const float alpha,
                               const float *A, const int lda, const int strideA,
                               const float *B, const int ldb, const int strideB,
                               const float beta,
                               float *C, const int ldc, const int strideC,
                               const int batch_count);

void fb_dgemm_strided_batched(const FB_LAYOUT Layout,
                               const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const double alpha,
                               const double *A, const int lda, const int strideA,
                               const double *B, const int ldb, const int strideB,
                               const double beta,
                               double *C, const int ldc, const int strideC,
                               const int batch_count);

void fb_cgemm_strided_batched(const FB_LAYOUT Layout,
                               const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const void *alpha,
                               const void *A, const int lda, const int strideA,
                               const void *B, const int ldb, const int strideB,
                               const void *beta,
                               void *C, const int ldc, const int strideC,
                               const int batch_count);

void fb_zgemm_strided_batched(const FB_LAYOUT Layout,
                               const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const void *alpha,
                               const void *A, const int lda, const int strideA,
                               const void *B, const int ldb, const int strideB,
                               const void *beta,
                               void *C, const int ldc, const int strideC,
                               const int batch_count);

/* ============================================================================
 * Mixed Precision Operations
 * ========================================================================= */

/**
 * @brief Data type enumeration for mixed precision
 */
typedef enum {
    FB_DTYPE_F16  = 0,   /**< IEEE 754 half precision (16-bit) */
    FB_DTYPE_BF16 = 1,   /**< bfloat16 (16-bit) */
    FB_DTYPE_F32  = 2,   /**< IEEE 754 single precision (32-bit) */
    FB_DTYPE_F64  = 3,   /**< IEEE 754 double precision (64-bit) */
    FB_DTYPE_I8   = 4,   /**< Signed 8-bit integer */
    FB_DTYPE_I32  = 5    /**< Signed 32-bit integer */
} FB_DTYPE;

/**
 * @brief Mixed precision GEMM with explicit data types
 * 
 * Allows different precisions for input, computation, and output:
 *   C = alpha * A * B + beta * C
 * 
 * Common use cases:
 * - Input: FP16, Compute: FP32, Output: FP16 (Tensor Core acceleration)
 * - Input: INT8, Compute: INT32, Output: FP32 (Quantized inference)
 * - Input: BF16, Compute: FP32, Output: FP32 (Training with mixed precision)
 * 
 * @param Layout Matrix layout
 * @param TransA Transpose operation for A
 * @param TransB Transpose operation for B
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 * @param input_dtype Data type of A and B matrices
 * @param compute_dtype Data type for internal computation
 * @param output_dtype Data type of C matrix
 * @param alpha Scaling factor (type matches compute_dtype)
 * @param A Input matrix A
 * @param lda Leading dimension of A
 * @param B Input matrix B
 * @param ldb Leading dimension of B
 * @param beta Scaling factor (type matches compute_dtype)
 * @param C Output matrix C
 * @param ldc Leading dimension of C
 */
void fb_gemm_ex(const FB_LAYOUT Layout,
                const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                const int M, const int N, const int K,
                const FB_DTYPE input_dtype,
                const FB_DTYPE compute_dtype,
                const FB_DTYPE output_dtype,
                const void *alpha,
                const void *A, const int lda,
                const void *B, const int ldb,
                const void *beta,
                void *C, const int ldc);

/* ============================================================================
 * Fused Operations
 * ========================================================================= */

/**
 * @brief Fused GEMM + activation: C = activation(alpha * A * B + beta * C)
 * 
 * Activation types supported
 */
typedef enum {
    FB_ACTIVATION_NONE    = 0,  /**< No activation (standard GEMM) */
    FB_ACTIVATION_RELU    = 1,  /**< ReLU: max(0, x) */
    FB_ACTIVATION_SIGMOID = 2,  /**< Sigmoid: 1 / (1 + exp(-x)) */
    FB_ACTIVATION_TANH    = 3,  /**< Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) */
    FB_ACTIVATION_GELU    = 4,  /**< GELU: x * Φ(x) */
    FB_ACTIVATION_SWISH   = 5   /**< Swish: x * sigmoid(x) */
} FB_ACTIVATION;

/**
 * @brief GEMM with fused activation function
 * 
 * Reduces memory bandwidth by computing activation in-place during GEMM
 * instead of requiring a separate pass over the output matrix.
 */
void fb_sgemm_activation(const FB_LAYOUT Layout,
                         const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                         const int M, const int N, const int K,
                         const float alpha,
                         const float *A, const int lda,
                         const float *B, const int ldb,
                         const float beta,
                         float *C, const int ldc,
                         const FB_ACTIVATION activation);

void fb_dgemm_activation(const FB_LAYOUT Layout,
                         const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                         const int M, const int N, const int K,
                         const double alpha,
                         const double *A, const int lda,
                         const double *B, const int ldb,
                         const double beta,
                         double *C, const int ldc,
                         const FB_ACTIVATION activation);

/**
 * @brief Fused GEMM + bias + activation: C = activation(alpha * A * B + bias)
 * 
 * @param bias Vector of length N (broadcast across rows) or M (broadcast across columns)
 * @param bias_mode 0 = row-wise (bias length N), 1 = column-wise (bias length M)
 */
void fb_sgemm_bias_activation(const FB_LAYOUT Layout,
                               const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const float alpha,
                               const float *A, const int lda,
                               const float *B, const int ldb,
                               const float *bias, const int bias_mode,
                               float *C, const int ldc,
                               const FB_ACTIVATION activation);

void fb_dgemm_bias_activation(const FB_LAYOUT Layout,
                               const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const double alpha,
                               const double *A, const int lda,
                               const double *B, const int ldb,
                               const double *bias, const int bias_mode,
                               double *C, const int ldc,
                               const FB_ACTIVATION activation);

/* ============================================================================
 * Operation Chains
 * ========================================================================= */

/**
 * @brief Opaque handle to a compiled operation chain
 * 
 * Operation chains allow multiple BLAS operations to be fused together
 * and optimized as a unit, reducing memory traffic and kernel launch overhead.
 */
typedef struct fb_op_chain_t fb_op_chain_t;

/**
 * @brief Create a new operation chain
 * 
 * @return Chain handle, or NULL on failure
 */
fb_op_chain_t* fb_op_chain_create(void);

/**
 * @brief Add a GEMM operation to the chain
 * 
 * @param chain Chain handle
 * @param name Symbolic name for the operation's output (used by later ops)
 * @param TransA Transpose for A
 * @param TransB Transpose for B
 * @param M Rows
 * @param N Columns
 * @param K Inner dimension
 * @param input_A Name of input A (or NULL for user-provided matrix)
 * @param input_B Name of input B (or NULL for user-provided matrix)
 * @return 0 on success, negative on error
 */
int fb_op_chain_add_gemm(fb_op_chain_t *chain, const char *name,
                         FB_TRANSPOSE TransA, FB_TRANSPOSE TransB,
                         int M, int N, int K,
                         const char *input_A, const char *input_B);

/**
 * @brief Compile the operation chain
 * 
 * Analyzes the chain and generates an optimized execution plan.
 * Must be called before fb_op_chain_execute().
 * 
 * @param chain Chain handle
 * @return 0 on success, negative on error
 */
int fb_op_chain_compile(fb_op_chain_t *chain);

/**
 * @brief Execute the operation chain
 * 
 * @param chain Compiled chain handle
 * @param inputs Array of input pointers (ordered by add operations)
 * @param outputs Array of output pointers
 * @return 0 on success, negative on error
 */
int fb_op_chain_execute(fb_op_chain_t *chain, void **inputs, void **outputs);

/**
 * @brief Destroy an operation chain
 * 
 * @param chain Chain handle to destroy
 */
void fb_op_chain_destroy(fb_op_chain_t *chain);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_EXT_H */
