/**
 * @file test_cublas_basic.c
 * @brief Basic functionality test for cuBLAS backend trait implementation
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
        printf("PASS: %s\n", msg); \
    } while(0)

#define EPSILON 1e-5f

// External trait reference (defined in cublas_trait_impl.c)
extern const fb_gpu_backend_trait_t fb_cublas_trait;

int test_lifecycle() {
    printf("\n=== Testing Lifecycle ===\n");
    
    void* handle = NULL;
    int result = fb_cublas_trait.init(0, NULL, &handle);
    TEST_ASSERT(result == 0, "cuBLAS initialization");
    TEST_ASSERT(handle != NULL, "Handle is not NULL");
    
    fb_cublas_trait.shutdown(handle);
    printf("PASS: Shutdown completed\n");
    
    return 0;
}

int test_device_properties() {
    printf("\n=== Testing Device Properties ===\n");
    
    void* handle = NULL;
    fb_cublas_trait.init(0, NULL, &handle);
    
    char name[256];
    size_t total_mem;
    int result = fb_cublas_trait.get_device_properties(handle, 0, name, sizeof(name), &total_mem);
    
    TEST_ASSERT(result == 0, "Get device properties");
    printf("Device: %s\n", name);
    printf("Memory: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    
    fb_cublas_trait.shutdown(handle);
    return 0;
}

int test_memory_management() {
    printf("\n=== Testing Memory Management ===\n");
    
    void* handle = NULL;
    fb_cublas_trait.init(0, NULL, &handle);
    
    // Allocate device memory
    size_t size = 1000 * sizeof(float);
    fb_gpu_ptr_t d_mem;
    int ret = fb_cublas_trait.malloc(handle, &d_mem, size);
    TEST_ASSERT(ret == 0 && d_mem != NULL, "Device memory allocation");
    
    // Allocate host memory
    float* h_data = (float*)malloc(size);
    for (int i = 0; i < 1000; i++) {
        h_data[i] = (float)i;
    }
    
    // Test H2D transfer
    fb_cublas_trait.memcpy_h2d(handle, d_mem, h_data, size);
    printf("PASS: Host to device transfer\n");
    
    // Test D2H transfer
    float* h_result = (float*)malloc(size);
    fb_cublas_trait.memcpy_d2h(handle, h_result, d_mem, size);
    printf("PASS: Device to host transfer\n");
    
    // Verify data
    int mismatch = 0;
    for (int i = 0; i < 1000; i++) {
        if (fabsf(h_result[i] - h_data[i]) > EPSILON) {
            mismatch++;
        }
    }
    TEST_ASSERT(mismatch == 0, "Data integrity after H2D and D2H");
    
    // Test D2D transfer
    fb_gpu_ptr_t d_mem2;
    fb_cublas_trait.malloc(handle, &d_mem2, size);
    fb_cublas_trait.memcpy_d2d(handle, d_mem2, d_mem, size);
    printf("PASS: Device to device transfer\n");
    
    // Cleanup
    fb_cublas_trait.free(handle, d_mem);
    fb_cublas_trait.free(handle, d_mem2);
    free(h_data);
    free(h_result);
    
    fb_cublas_trait.shutdown(handle);
    return 0;
}

int test_stream_operations() {
    printf("\n=== Testing Stream Operations ===\n");
    
    void* handle = NULL;
    fb_cublas_trait.init(0, NULL, &handle);
    
    // Create stream
    fb_gpu_stream_t stream;
    int ret = fb_cublas_trait.stream_create(handle, &stream);
    TEST_ASSERT(ret == 0 && stream != NULL, "Stream creation");
    
    // Synchronize stream
    fb_cublas_trait.stream_synchronize(handle, stream);
    printf("PASS: Stream synchronization\n");
    
    // Destroy stream
    fb_cublas_trait.stream_destroy(handle, stream);
    printf("PASS: Stream destruction\n");
    
    fb_cublas_trait.shutdown(handle);
    return 0;
}

int test_saxpy() {
    printf("\n=== Testing SAXPY (Level 1 BLAS) ===\n");
    
    void* handle = NULL;
    fb_cublas_trait.init(0, NULL, &handle);
    
    int n = 1000;
    float alpha = 2.5f;
    
    // Allocate and initialize host data
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(n * sizeof(float));
    float* h_y_expected = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(n - i);
        h_y_expected[i] = alpha * h_x[i] + h_y[i]; // y = alpha*x + y
    }
    
    // Allocate device memory
    fb_gpu_ptr_t d_x, d_y;
    fb_cublas_trait.malloc(handle, &d_x, n * sizeof(float));
    fb_cublas_trait.malloc(handle, &d_y, n * sizeof(float));
    
    // Transfer to device
    fb_cublas_trait.memcpy_h2d(handle, d_x, h_x, n * sizeof(float));
    fb_cublas_trait.memcpy_h2d(handle, d_y, h_y, n * sizeof(float));
    
    // Execute SAXPY
    fb_cublas_trait.saxpy(handle, NULL, n, alpha, d_x, 1, d_y, 1);
    
    // Transfer result back
    float* h_y_result = (float*)malloc(n * sizeof(float));
    fb_cublas_trait.memcpy_d2h(handle, h_y_result, d_y, n * sizeof(float));
    
    // Verify result
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_y_result[i] - h_y_expected[i]) > EPSILON) {
            errors++;
            if (errors <= 5) {
                printf("Error at %d: got %f, expected %f\n", i, h_y_result[i], h_y_expected[i]);
            }
        }
    }
    TEST_ASSERT(errors == 0, "SAXPY correctness");
    
    // Cleanup
    fb_cublas_trait.free(handle, d_x);
    fb_cublas_trait.free(handle, d_y);
    free(h_x);
    free(h_y);
    free(h_y_expected);
    free(h_y_result);
    
    fb_cublas_trait.shutdown(handle);
    return 0;
}

int test_sgemv() {
    printf("\n=== Testing SGEMV (Level 2 BLAS) ===\n");
    
    void* handle = NULL;
    fb_cublas_trait.init(0, NULL, &handle);
    
    int m = 100, n = 100;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host data
    float* h_A = (float*)malloc(m * n * sizeof(float));
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(m * sizeof(float));
    float* h_y_expected = (float*)malloc(m * sizeof(float));
    
    // Initialize: A = identity, x = [1, 2, 3, ...], y = 0
    for (int i = 0; i < m * n; i++) {
        h_A[i] = (i / n == i % n) ? 1.0f : 0.0f; // Identity matrix
    }
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)(i + 1);
    }
    for (int i = 0; i < m; i++) {
        h_y[i] = 0.0f;
        h_y_expected[i] = (float)(i + 1); // Identity * x = x
    }
    
    // Allocate device memory
    fb_gpu_ptr_t d_A, d_x, d_y;
    fb_cublas_trait.malloc(handle, &d_A, m * n * sizeof(float));
    fb_cublas_trait.malloc(handle, &d_x, n * sizeof(float));
    fb_cublas_trait.malloc(handle, &d_y, m * sizeof(float));
    
    // Transfer to device
    fb_cublas_trait.memcpy_h2d(handle, d_A, h_A, m * n * sizeof(float));
    fb_cublas_trait.memcpy_h2d(handle, d_x, h_x, n * sizeof(float));
    fb_cublas_trait.memcpy_h2d(handle, d_y, h_y, m * sizeof(float));
    
    // Execute SGEMV: y = alpha*A*x + beta*y
    fb_cublas_trait.sgemv(handle, NULL, 'N', m, n, alpha, d_A, m, d_x, 1, beta, d_y, 1);
    
    // Transfer result back
    float* h_y_result = (float*)malloc(m * sizeof(float));
    fb_cublas_trait.memcpy_d2h(handle, h_y_result, d_y, m * sizeof(float));
    
    // Verify result
    int errors = 0;
    for (int i = 0; i < m; i++) {
        if (fabsf(h_y_result[i] - h_y_expected[i]) > EPSILON) {
            errors++;
            if (errors <= 5) {
                printf("Error at %d: got %f, expected %f\n", i, h_y_result[i], h_y_expected[i]);
            }
        }
    }
    TEST_ASSERT(errors == 0, "SGEMV correctness");
    
    // Cleanup
    fb_cublas_trait.free(handle, d_A);
    fb_cublas_trait.free(handle, d_x);
    fb_cublas_trait.free(handle, d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y_expected);
    free(h_y_result);
    
    fb_cublas_trait.shutdown(handle);
    return 0;
}

int test_sgemm() {
    printf("\n=== Testing SGEMM (Level 3 BLAS) ===\n");
    
    void* handle = NULL;
    fb_cublas_trait.init(0, NULL, &handle);
    
    int m = 64, n = 64, k = 64;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host data
    float* h_A = (float*)malloc(m * k * sizeof(float));
    float* h_B = (float*)malloc(k * n * sizeof(float));
    float* h_C = (float*)malloc(m * n * sizeof(float));
    
    // Initialize: A = I, B = I, C should = I
    for (int i = 0; i < m * k; i++) {
        h_A[i] = (i / k == i % k) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < k * n; i++) {
        h_B[i] = (i / n == i % n) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < m * n; i++) {
        h_C[i] = 0.0f;
    }
    
    // Allocate device memory
    fb_gpu_ptr_t d_A, d_B, d_C;
    fb_cublas_trait.malloc(handle, &d_A, m * k * sizeof(float));
    fb_cublas_trait.malloc(handle, &d_B, k * n * sizeof(float));
    fb_cublas_trait.malloc(handle, &d_C, m * n * sizeof(float));
    
    // Transfer to device
    fb_cublas_trait.memcpy_h2d(handle, d_A, h_A, m * k * sizeof(float));
    fb_cublas_trait.memcpy_h2d(handle, d_B, h_B, k * n * sizeof(float));
    fb_cublas_trait.memcpy_h2d(handle, d_C, h_C, m * n * sizeof(float));
    
    // Execute SGEMM: C = alpha*A*B + beta*C
    fb_cublas_trait.sgemm(handle, NULL, 'N', 'N', m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m);
    
    // Transfer result back
    float* h_C_result = (float*)malloc(m * n * sizeof(float));
    fb_cublas_trait.memcpy_d2h(handle, h_C_result, d_C, m * n * sizeof(float));
    
    // Verify result (should be identity)
    int errors = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float got = h_C_result[i + j * m];
            if (fabsf(got - expected) > EPSILON) {
                errors++;
                if (errors <= 5) {
                    printf("Error at (%d,%d): got %f, expected %f\n", i, j, got, expected);
                }
            }
        }
    }
    TEST_ASSERT(errors == 0, "SGEMM correctness");
    
    // Cleanup
    fb_cublas_trait.free(handle, d_A);
    fb_cublas_trait.free(handle, d_B);
    fb_cublas_trait.free(handle, d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_result);
    
    fb_cublas_trait.shutdown(handle);
    return 0;
}

int main() {
    printf("=================================================\n");
    printf("cuBLAS Backend Trait - Basic Functionality Tests\n");
    printf("=================================================\n");
    
    int result = 0;
    
    result |= test_lifecycle();
    result |= test_device_properties();
    result |= test_memory_management();
    result |= test_stream_operations();
    result |= test_saxpy();
    result |= test_sgemv();
    result |= test_sgemm();
    
    printf("\n=================================================\n");
    if (result == 0) {
        printf("ALL TESTS PASSED ✓\n");
    } else {
        printf("SOME TESTS FAILED ✗\n");
    }
    printf("=================================================\n");
    
    return result;
}

