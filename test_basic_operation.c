#include "faster_blaster.h"
#include "src/core/dispatch.h"
#include "src/backends/blis_backend.h"
#include "src/backends/aocl_backend.h"
#include "src/backends/gpu/cublas_backend.h"
#include <stdio.h>
#

int main() {
    printf("=== Basic BLAS Operation Test ===\n");
    printf("Testing simple saxpy operation with real backends\n\n");
    
    // Initialize dispatch system
    printf("Initializing dispatch system...\n");
    fb_dispatch_init_global();
    fb_dispatch_table_t* dispatch = fb_dispatch_global();
    
    // Register BLIS backend
    printf("Registering BLIS backend...\n");
    const fb_backend_vtable_t* blis_vtable = fb_blis_get_vtable();
    if (blis_vtable) {
        fb_dispatch_register_backend(dispatch, "BLIS", blis_vtable);
        printf("  ✓ BLIS registered\n");
    } else {
        printf("  ✗ BLIS not available\n");
        return 1;
    }
    
    // Register AOCL backend
    printf("Registering AOCL backend...\n");
    const fb_backend_vtable_t* aocl_vtable = fb_aocl_get_vtable();
    if (aocl_vtable) {
        fb_dispatch_register_backend(dispatch, "AOCL", aocl_vtable);
        printf("  ✓ AOCL registered\n");
    }
    
    // Select first backend
    fb_dispatch_auto_select(dispatch);
    printf("\nActive backend: %s\n", fb_dispatch_get_active_backend_name(dispatch));
    
    // Simple saxpy test: y = y + 2.0 * x
    printf("\nTesting saxpy: y = y + 2.0*x\n");
    int n = 5;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    printf("Input x:  [1, 2, 3, 4, 5]\n");
    printf("Input y:  [0, 0, 0, 0, 0]\n");
    
    // Call saxpy through dispatch
    fb_saxpy(n, 2.0f, x, 1, y, 1);
    
    printf("Output y: [");
    for (int i = 0; i < n; i++) {
        printf("%.0f%s", y[i], i < n-1 ? ", " : "");
    }
    printf("]\n");
    
    // Verify results
    printf("\nVerification:\n");
    int passed = 1;
    for (int i = 0; i < n; i++) {
        float expected = 2.0f * (i + 1);
        if (fabsf(y[i] - expected) < 0.001f) {
            printf("  y[%d] = %.0f ✓\n", i, y[i]);
        } else {
            printf("  y[%d] = %.0f ✗ (expected %.0f)\n", i, y[i], expected);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("\n✅ Operation executed correctly on backend!\n");
        return 0;
    } else {
        printf("\n❌ Operation results incorrect\n");
        return 1;
    }
}
