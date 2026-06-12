# Example: Creating a Custom Backend Plugin

This example demonstrates how to create a custom BLAS backend plugin for faster-blaster.

## Scenario

We'll create a simple custom backend that:
1. Implements a subset of BLAS Level 1 operations
2. Uses SIMD intrinsics for optimization
3. Registers with the backend loader
4. Can be selected at runtime

## File Structure

```
my_custom_backend/
├── include/
│   └── my_backend.h
├── src/
│   ├── my_backend.c
│   ├── my_level1.c
│   └── my_simd.c
├── CMakeLists.txt
└── README.md
```

## Implementation

### 1. Header File (`my_backend.h`)

```c
#ifndef MY_CUSTOM_BACKEND_H
#define MY_CUSTOM_BACKEND_H

#include "backend_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the custom backend vtable
 */
const fb_backend_vtable_t* my_backend_get_vtable(void);

/**
 * Initialize custom backend
 */
int my_backend_init(void);

/**
 * Shutdown custom backend
 */
void my_backend_shutdown(void);

/**
 * Check if custom backend is available
 */
bool my_backend_is_available(void);

#ifdef __cplusplus
}
#endif

#endif /* MY_CUSTOM_BACKEND_H */
```

### 2. Main Backend File (`my_backend.c`)

```c
#include "my_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Forward declarations */
extern void my_sasum(int n, const float* x, int incx, float* result);
extern void my_dasum(int n, const double* x, int incx, double* result);
extern void my_saxpy(int n, float alpha, const float* x, int incx, 
                     float* y, int incy);
extern void my_daxpy(int n, double alpha, const double* x, int incx, 
                     double* y, int incy);
extern void my_sdot(int n, const float* x, int incx, 
                    const float* y, int incy, float* result);
extern void my_ddot(int n, const double* x, int incx, 
                    const double* y, int incy, double* result);
extern void my_scopy(int n, const float* x, int incx, 
                     float* y, int incy);
extern void my_dcopy(int n, const double* x, int incx, 
                     double* y, int incy);
extern void my_sscal(int n, float alpha, float* x, int incx);
extern void my_dscal(int n, double alpha, double* x, int incx);

/* Global state */
static bool g_initialized = false;

/* Backend vtable */
static fb_backend_vtable_t g_my_vtable = {
    /* Level 1 BLAS - Only implement a subset */
    .sasum = my_sasum,
    .dasum = my_dasum,
    .saxpy = my_saxpy,
    .daxpy = my_daxpy,
    .sdot = my_sdot,
    .ddot = my_ddot,
    .scopy = my_scopy,
    .dcopy = my_dcopy,
    .sscal = my_sscal,
    .dscal = my_dscal,
    
    /* Leave other operations NULL - will fallback to reference backend */
    /* Or populate with reference backend functions */
};

const fb_backend_vtable_t* my_backend_get_vtable(void) {
    return &g_my_vtable;
}

int my_backend_init(void) {
    if (g_initialized) {
        return 0;
    }
    
    /* Perform any initialization needed */
    /* Check CPU features, allocate resources, etc. */
    
    g_initialized = true;
    return 0;
}

void my_backend_shutdown(void) {
    if (!g_initialized) {
        return;
    }
    
    /* Cleanup resources */
    
    g_initialized = false;
}

bool my_backend_is_available(void) {
    /* Check if this backend can run on this system */
    /* E.g., check for required CPU features */
    
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
    return true;  /* Requires SSE2 */
#else
    return false;
#endif
}
```

### 3. Level 1 Operations (`my_level1.c`)

```c
#include "my_backend.h"
#
#include <stddef.h>

/* Basic implementations without SIMD (fallback) */

void my_sasum(int n, const float* x, int incx, float* result) {
    if (n <= 0 || incx <= 0) {
        *result = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(x[i * incx]);
    }
    *result = sum;
}

void my_dasum(int n, const double* x, int incx, double* result) {
    if (n <= 0 || incx <= 0) {
        *result = 0.0;
        return;
    }
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs(x[i * incx]);
    }
    *result = sum;
}

void my_saxpy(int n, float alpha, const float* x, int incx, 
              float* y, int incy) {
    if (n <= 0) {
        return;
    }
    
    if (incx == 1 && incy == 1) {
        /* Optimized for contiguous memory */
        for (int i = 0; i < n; i++) {
            y[i] += alpha * x[i];
        }
    } else {
        /* General case */
        for (int i = 0; i < n; i++) {
            y[i * incy] += alpha * x[i * incx];
        }
    }
}

void my_daxpy(int n, double alpha, const double* x, int incx, 
              double* y, int incy) {
    if (n <= 0) {
        return;
    }
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            y[i] += alpha * x[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            y[i * incy] += alpha * x[i * incx];
        }
    }
}

void my_sdot(int n, const float* x, int incx, 
             const float* y, int incy, float* result) {
    if (n <= 0) {
        *result = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i * incx] * y[i * incy];
    }
    *result = sum;
}

void my_ddot(int n, const double* x, int incx, 
             const double* y, int incy, double* result) {
    if (n <= 0) {
        *result = 0.0;
        return;
    }
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i * incx] * y[i * incy];
    }
    *result = sum;
}

void my_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (n <= 0) {
        return;
    }
    
    for (int i = 0; i < n; i++) {
        y[i * incy] = x[i * incx];
    }
}

void my_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (n <= 0) {
        return;
    }
    
    for (int i = 0; i < n; i++) {
        y[i * incy] = x[i * incx];
    }
}

void my_sscal(int n, float alpha, float* x, int incx) {
    if (n <= 0 || incx <= 0) {
        return;
    }
    
    for (int i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}

void my_dscal(int n, double alpha, double* x, int incx) {
    if (n <= 0 || incx <= 0) {
        return;
    }
    
    for (int i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}
```

### 4. SIMD-Optimized Versions (`my_simd.c`)

```c
#include "my_backend.h"

#if defined(__SSE2__)
#include <emmintrin.h>  /* SSE2 */
#endif

#if defined(__AVX__)
#include <immintrin.h>  /* AVX */
#endif

/* SIMD-optimized dot product for contiguous arrays */
#if defined(__AVX__)

void my_sdot_avx(int n, const float* x, const float* y, float* result) {
    __m256 sum_vec = _mm256_setzero_ps();
    
    /* Process 8 elements at a time */
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);
        __m256 prod = _mm256_mul_ps(x_vec, y_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod);
    }
    
    /* Horizontal sum of vector */
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum = _mm_add_ps(sum_low, sum_high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    
    float scalar_sum = _mm_cvtss_f32(sum);
    
    /* Handle remaining elements */
    for (; i < n; i++) {
        scalar_sum += x[i] * y[i];
    }
    
    *result = scalar_sum;
}

#elif defined(__SSE2__)

void my_sdot_sse2(int n, const float* x, const float* y, float* result) {
    __m128 sum_vec = _mm_setzero_ps();
    
    /* Process 4 elements at a time */
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(&x[i]);
        __m128 y_vec = _mm_loadu_ps(&y[i]);
        __m128 prod = _mm_mul_ps(x_vec, y_vec);
        sum_vec = _mm_add_ps(sum_vec, prod);
    }
    
    /* Horizontal sum */
    __m128 sum1 = _mm_hadd_ps(sum_vec, sum_vec);
    __m128 sum2 = _mm_hadd_ps(sum1, sum1);
    float scalar_sum = _mm_cvtss_f32(sum2);
    
    /* Handle remaining elements */
    for (; i < n; i++) {
        scalar_sum += x[i] * y[i];
    }
    
    *result = scalar_sum;
}

#endif

/* SIMD-optimized SAXPY */
#if defined(__AVX__)

void my_saxpy_avx(int n, float alpha, const float* x, float* y) {
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);
        __m256 result = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);
        _mm256_storeu_ps(&y[i], result);
    }
    
    /* Handle remaining elements */
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

#endif
```

### 5. Registration Example

```c
/* example_usage.c */
#include "backend_loader.h"
#include "my_backend.h"
#include <stdio.h>

int main(void) {
    /* Initialize backend loader */
    fb_backend_loader_init();
    
    /* Check if custom backend is available */
    if (my_backend_is_available()) {
        printf("Custom backend is available\n");
        
        /* Initialize custom backend */
        my_backend_init();
        
        /* Register it */
        uint32_t caps = FB_CAP_CPU | FB_CAP_LEVEL1 | 
                        FB_CAP_SINGLE | FB_CAP_DOUBLE;
        
        fb_backend_register_custom("MyOptimizedBackend",
                                   my_backend_get_vtable(),
                                   caps);
        
        /* List available backends */
        fb_backend_info_t backends[FB_BACKEND_COUNT];
        int count = fb_backend_list_available(backends, FB_BACKEND_COUNT);
        
        printf("Available backends:\n");
        for (int i = 0; i < count; i++) {
            printf("  - %s (priority: %d)\n", 
                   backends[i].name, backends[i].priority);
        }
        
        /* Use custom backend */
        fb_backend_set_current(FB_BACKEND_CUSTOM);
        
        /* Now all operations use custom backend */
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float result;
        
        fb_sdot(4, x, 1, y, 1, &result);
        printf("dot product: %f\n", result);
        
        /* Cleanup */
        my_backend_shutdown();
    } else {
        printf("Custom backend not available on this system\n");
    }
    
    fb_backend_loader_shutdown();
    return 0;
}
```

### 6. CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_custom_backend C)

# Check for SIMD support
include(CheckCCompilerFlag)

check_c_compiler_flag(-mavx HAS_AVX)
check_c_compiler_flag(-msse2 HAS_SSE2)

# Build custom backend library
add_library(my_backend STATIC
    src/my_backend.c
    src/my_level1.c
    src/my_simd.c
)

target_include_directories(my_backend PUBLIC
    include
    ${FASTER_BLASTER_INCLUDE_DIR}
)

# Add SIMD flags if supported
if(HAS_AVX)
    target_compile_options(my_backend PRIVATE -mavx)
    target_compile_definitions(my_backend PRIVATE __AVX__)
elseif(HAS_SSE2)
    target_compile_options(my_backend PRIVATE -msse2)
    target_compile_definitions(my_backend PRIVATE __SSE2__)
endif()

# Build example
add_executable(backend_example
    example_usage.c
)

target_link_libraries(backend_example
    my_backend
    faster_blaster
)
```

## Building and Running

```powershell
# Build custom backend
cd my_custom_backend
mkdir build
cd build
cmake .. -DFASTER_BLASTER_INCLUDE_DIR=../../faster-blaster/src
cmake --build .

# Run example
./backend_example
```

## Expected Output

```
Custom backend is available
Available backends:
  - Reference (priority: 1)
  - OpenBLAS (priority: 50)
  - MyOptimizedBackend (priority: 60)
dot product: 70.000000
```

## Performance Comparison

Add benchmarking to compare backends:

```c
#include <time.h>

void benchmark_sdot(int n, int iterations) {
    float* x = malloc(n * sizeof(float));
    float* y = malloc(n * sizeof(float));
    
    /* Initialize arrays */
    for (int i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = (float)(n - i);
    }
    
    clock_t start = clock();
    float result;
    for (int iter = 0; iter < iterations; iter++) {
        fb_sdot(n, x, 1, y, 1, &result);
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double flops = (double)n * 2.0 * iterations / elapsed / 1e9;
    
    printf("n=%d, GFLOPS: %.2f\n", n, flops);
    
    free(x);
    free(y);
}

int main(void) {
    fb_backend_loader_init();
    
    /* Benchmark reference backend */
    printf("Reference Backend:\n");
    fb_backend_set_current(FB_BACKEND_REFERENCE);
    benchmark_sdot(1000000, 1000);
    
    /* Benchmark custom backend */
    if (my_backend_is_available()) {
        printf("\nCustom Backend:\n");
        my_backend_init();
        fb_backend_register_custom("Custom", my_backend_get_vtable(), 
                                    FB_CAP_CPU | FB_CAP_LEVEL1);
        fb_backend_set_current(FB_BACKEND_CUSTOM);
        benchmark_sdot(1000000, 1000);
        my_backend_shutdown();
    }
    
    fb_backend_loader_shutdown();
    return 0;
}
```

This example demonstrates a complete custom backend implementation with SIMD optimizations, proper registration, and benchmarking capabilities.
