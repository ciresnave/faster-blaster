/**
 * @file test_level1_ops.c
 * @brief Focused test for Level 1 operations (sdot, snrm2)
 */

#include "faster-blaster/device_memory_manager.h"
#include "faster-blaster/gpu_backend_trait.h"
#include "faster_blaster.h"
#include <stdio.h>
#include <math.h>

extern int fb_cublas_smart_init(int device_id, void* lib_handle);
extern void fb_cublas_smart_shutdown(void);
extern float cublas_sdot_smart_wrapper(int n, const float* x, int incx, const float* y, int incy);
extern float cublas_snrm2_smart_wrapper(int n, const float* x, int incx);

int main(void) {
    printf("=== Level 1 Operations Test ===\n\n");
    
    fb_cublas_smart_init(0, NULL);
    
    // Test sdot in isolation
    printf("Test 1: sdot\n");
    float x1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y1[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    printf("x = [1, 2, 3, 4, 5]\n");
    printf("y = [2, 3, 4, 5, 6]\n");
    printf("Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2+6+12+20+30 = 70\n");
    
    float dot = cublas_sdot_smart_wrapper(5, x1, 1, y1, 1);
    printf("Result: %.4f\n", dot);
    printf("%s\n\n", fabsf(dot - 70.0f) < 1e-5f ? "✅ PASS" : "❌ FAIL");
    
    // Test snrm2 in isolation (fresh arrays)
    printf("Test 2: snrm2\n");
    float x2[] = {3.0f, 4.0f, 0.0f};
    
    printf("x = [3, 4, 0]\n");
    printf("Expected: sqrt(9 + 16 + 0) = sqrt(25) = 5\n");
    
    float norm = cublas_snrm2_smart_wrapper(3, x2, 1);
    printf("Result: %.4f\n", norm);
    printf("%s\n", fabsf(norm - 5.0f) < 1e-5f ? "✅ PASS" : "❌ FAIL");
    
    fb_cublas_smart_shutdown();
    return 0;
}
