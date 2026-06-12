/**
 * @file simple_gemm.c
 * @brief Simple GEMM example
 * 
 * Demonstrates basic usage of faster-blaster library with matrix multiplication.
 */

#include "faster_blaster.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /* Initialize library */
    if (fb_init() != 0) {
        fprintf(stderr, "Failed to initialize faster-blaster\n");
        return 1;
    }
    
    /* Matrix dimensions: C = A * B, where A is 3x4, B is 4x2, C is 3x2 */
    const int64_t m = 3;
    const int64_t n = 2;
    const int64_t k = 4;
    
    /* Allocate and initialize matrices */
    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *C = calloc(m * n, sizeof(float));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A); free(B); free(C);
        fb_finalize();
        return 1;
    }
    
    /* Initialize A (row-major) */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++) {
            A[i * k + j] = (float)(i * k + j + 1);
        }
    }
    
    /* Initialize B (row-major) */
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < n; j++) {
            B[i * n + j] = (float)(i * n + j + 1);
        }
    }
    
    /* Print input matrices */
    printf("Matrix A (%lld x %lld):\n", (long long)m, (long long)k);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++) {
            printf("%6.1f ", A[i * k + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Matrix B (%lld x %lld):\n", (long long)k, (long long)n);
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < n; j++) {
            printf("%6.1f ", B[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    /* Perform matrix multiplication: C = A * B */
    fb_sgemm(
        FbRowMajor,  /* layout */
        FbNoTrans,           /* no transpose on A */
        FbNoTrans,           /* no transpose on B */
        m, n, k,               /* dimensions */
        1.0f,                  /* alpha = 1.0 */
        A, k,                  /* A and lda */
        B, n,                  /* B and ldb */
        0.0f,                  /* beta = 0.0 */
        C, n                   /* C and ldc */
    );
    
    /* Print result */
    printf("Matrix C = A * B (%lld x %lld):\n", (long long)m, (long long)n);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            printf("%6.1f ", C[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    /* Example with alpha and beta scaling: C = 2.0*A*B + 1.0*C */
    printf("Computing C = 2.0*A*B + 1.0*C (with existing C):\n");
    fb_sgemm(
        FbRowMajor,
        FbNoTrans, FbNoTrans,
        m, n, k,
        2.0f,  /* alpha = 2.0 */
        A, k,
        B, n,
        1.0f,  /* beta = 1.0 (accumulate with existing C) */
        C, n
    );
    
    printf("Result:\n");
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            printf("%6.1f ", C[i * n + j]);
        }
        printf("\n");
    }
    
    /* Cleanup */
    free(A);
    free(B);
    free(C);
    
    fb_finalize();
    
    printf("\nExample completed successfully!\n");
    return 0;
}
