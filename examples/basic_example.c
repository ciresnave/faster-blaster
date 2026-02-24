/**
 * @file basic_example.c
 * @brief Basic usage example for Faster-BLASTER
 * 
 * Demonstrates:
 * - Library initialization
 * - Simple matrix multiplication
 * - Vector operations
 * - Cleanup
 */

#include <faster_blaster.h>
#include <stdio.h>
#include <stdlib.h>

void print_matrix(const char *name, const float *matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    printf("=== Faster-BLASTER Basic Example ===\n\n");
    
    // Initialize library
    printf("Initializing Faster-BLASTER...\n");
    if (fb_init(NULL) != 0) {
        fprintf(stderr, "Failed to initialize faster-blaster\n");
        return 1;
    }
    printf("Library version: %s\n", fb_get_version());
    printf("Initialization complete!\n\n");
    
    // Example 1: Matrix multiplication
    printf("Example 1: Matrix Multiplication (SGEMM)\n");
    printf("Computing C = A * B where A, B, C are 3x3 matrices\n\n");
    
    float A[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    
    float B[9] = {
        9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f
    };
    
    float C[9] = {0.0f};
    
    print_matrix("Matrix A", A, 3, 3);
    print_matrix("Matrix B", B, 3, 3);
    
    // Perform C = 1.0 * A * B + 0.0 * C
    fb_sgemm(FbRowMajor, FbNoTrans, FbNoTrans,
             3, 3, 3,              // M, N, K
             1.0f,                 // alpha
             A, 3,                 // A, lda
             B, 3,                 // B, ldb
             0.0f,                 // beta
             C, 3);                // C, ldc
    
    print_matrix("Result C = A * B", C, 3, 3);
    
    // Example 2: Vector operations
    printf("Example 2: Vector Operations\n");
    printf("Computing Y = alpha * X + Y (SAXPY)\n\n");
    
    float X[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float Y[5] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float alpha = 2.0f;
    
    printf("X = [");
    for (int i = 0; i < 5; i++) printf("%.1f ", X[i]);
    printf("]\n");
    
    printf("Y = [");
    for (int i = 0; i < 5; i++) printf("%.1f ", Y[i]);
    printf("]\n");
    
    printf("alpha = %.1f\n\n", alpha);
    
    fb_saxpy(5, alpha, X, 1, Y, 1);
    
    printf("Result Y = %.1f * X + Y = [", alpha);
    for (int i = 0; i < 5; i++) printf("%.1f ", Y[i]);
    printf("]\n\n");
    
    // Example 3: Dot product
    printf("Example 3: Dot Product (SDOT)\n");
    
    float X2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float Y2[4] = {4.0f, 3.0f, 2.0f, 1.0f};
    
    printf("X = [");
    for (int i = 0; i < 4; i++) printf("%.1f ", X2[i]);
    printf("]\n");
    
    printf("Y = [");
    for (int i = 0; i < 4; i++) printf("%.1f ", Y2[i]);
    printf("]\n");
    
    float dot_result = fb_sdot(4, X2, 1, Y2, 1);
    
    printf("Dot product X · Y = %.1f\n\n", dot_result);
    
    // Example 4: Vector norm
    printf("Example 4: Vector 2-Norm (SNRM2)\n");
    
    float V[4] = {3.0f, 4.0f, 0.0f, 0.0f};
    
    printf("V = [");
    for (int i = 0; i < 4; i++) printf("%.1f ", V[i]);
    printf("]\n");
    
    float norm = fb_snrm2(4, V, 1);
    
    printf("||V||₂ = %.3f\n\n", norm);
    
    // Cleanup
    printf("Cleaning up...\n");
    fb_finalize();
    
    printf("Done!\n");
    return 0;
}
