/**
 * @file vector_ops.c
 * @brief BLAS Level 1 vector operations example
 * 
 * Demonstrates basic vector operations.
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
    
    const int64_t n = 5;
    
    /* Allocate vectors */
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x); free(y);
        fb_finalize();
        return 1;
    }
    
    /* Initialize vectors */
    for (int64_t i = 0; i < n; i++) {
        x[i] = (float)(i + 1);
        y[i] = (float)(10 + i);
    }
    
    printf("Initial vectors:\n");
    printf("x = [");
    for (int64_t i = 0; i < n; i++) {
        printf("%.1f%s", x[i], (i < n-1) ? ", " : "");
    }
    printf("]\n");
    
    printf("y = [");
    for (int64_t i = 0; i < n; i++) {
        printf("%.1f%s", y[i], (i < n-1) ? ", " : "");
    }
    printf("]\n\n");
    
    /* Dot product: result = x^T * y */
    float dot = fb_sdot(n, x, 1, y, 1);
    printf("Dot product (x^T * y) = %.1f\n\n", dot);
    
    /* L2 norm: ||x||_2 */
    float norm = fb_snrm2(n, x, 1);
    printf("L2 norm (||x||_2) = %.4f\n\n", norm);
    
    /* AXPY: y := alpha*x + y */
    float alpha = 2.0f;
    printf("Computing y := %.1f*x + y\n", alpha);
    fb_saxpy(n, alpha, x, 1, y, 1);
    
    printf("Result y = [");
    for (int64_t i = 0; i < n; i++) {
        printf("%.1f%s", y[i], (i < n-1) ? ", " : "");
    }
    printf("]\n\n");
    
    /* Scale: x := alpha*x */
    alpha = 0.5f;
    printf("Scaling x by %.1f\n", alpha);
    fb_sscal(n, alpha, x, 1);
    
    printf("Result x = [");
    for (int64_t i = 0; i < n; i++) {
        printf("%.2f%s", x[i], (i < n-1) ? ", " : "");
    }
    printf("]\n\n");
    
    /* Copy: y := x */
    printf("Copying x to y\n");
    fb_scopy(n, x, 1, y, 1);
    
    printf("Result y = [");
    for (int64_t i = 0; i < n; i++) {
        printf("%.2f%s", y[i], (i < n-1) ? ", " : "");
    }
    printf("]\n");
    
    /* Cleanup */
    free(x);
    free(y);
    
    fb_finalize();
    
    printf("\nExample completed successfully!\n");
    return 0;
}
