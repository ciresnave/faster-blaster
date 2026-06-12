/**
 * @file example_hybrid_dispatch.c
 * @brief Example: Hybrid CPU/GPU dispatch in action
 * 
 * This example demonstrates the power of faster-blaster's hybrid dispatch:
 * - Automatic device selection based on load and problem size
 * - Transparent CPU/GPU switching
 * - Load balancing across multiple GPUs
 */

#include <faster-blaster/dispatch_api.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /* ========================================================================
     * 1. Initialize Library
     * ====================================================================== */
    
    printf("=== Faster-Blaster Hybrid Dispatch Example ===\n\n");
    
    /* Initialize with default configuration */
    if (fb_init(NULL) != 0) {
        fprintf(stderr, "Failed to initialize faster-blaster\n");
        return 1;
    }
    
    printf("Library version: %s\n", fb_get_version());
    printf("Available devices:\n");
    fb_print_devices();
    printf("\n");
    
    /* ========================================================================
     * 2. Allocate Test Matrices
     * ====================================================================== */
    
    int m = 2048, n = 2048, k = 2048;
    
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));
    
    /* Initialize matrices with random data */
    for (int i = 0; i < m * k; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < m * n; i++) C[i] = 0.0f;
    
    /* ========================================================================
     * 3. AUTOMATIC DISPATCH - Let library choose optimal device
     * ====================================================================== */
    
    printf("=== AUTOMATIC DISPATCH ===\n");
    printf("Computing C = A * B (2048x2048)...\n");
    
    /* Library will:
     * - Check problem size (large → GPU preferred)
     * - Check device loads
     * - Check data locality
     * - Select optimal device automatically
     */
    fb_sgemm_auto('N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, m);
    
    /* See which device was selected */
    fb_selection_explanation_t explanation;
    if (fb_get_last_selection_explanation(&explanation) == 0) {
        printf("Selected device: %d\n", explanation.selected_device);
        printf("Reason: %s\n", explanation.reason);
        printf("Score: %.3f\n\n", explanation.selected_score.total_score);
    }
    
    /* ========================================================================
     * 4. EXPLICIT DISPATCH - Force specific device
     * ====================================================================== */
    
    printf("=== EXPLICIT DISPATCH ===\n");
    
    int device_count = fb_get_device_count();
    for (int dev = 0; dev < device_count; dev++) {
        const fb_compute_device_t* device = fb_get_device_info(dev);
        printf("Computing on %s...\n", fb_device_get_name(device));
        
        /* Force computation on this specific device */
        fb_sgemm_on_device(dev, 'N', 'N', m, n, k,
                          1.0f, A, m, B, k, 0.0f, C, m);
    }
    printf("\n");
    
    /* ========================================================================
     * 5. POLICY-BASED DISPATCH - Different strategies
     * ====================================================================== */
    
    printf("=== POLICY-BASED DISPATCH ===\n");
    
    /* Strategy 1: FASTEST (ignore load, use fastest device) */
    printf("Policy: FASTEST\n");
    fb_set_dispatch_policy(FB_POLICY_FASTEST);
    fb_sgemm_auto('N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, m);
    fb_get_last_selection_explanation(&explanation);
    printf("  → Selected: %d (%s)\n\n",
           explanation.selected_device, explanation.reason);
    
    /* Strategy 2: LOAD_BALANCED (balance across all devices) */
    printf("Policy: LOAD_BALANCED\n");
    fb_set_dispatch_policy(FB_POLICY_LOAD_BALANCED);
    fb_sgemm_auto('N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, m);
    fb_get_last_selection_explanation(&explanation);
    printf("  → Selected: %d (%s)\n\n",
           explanation.selected_device, explanation.reason);
    
    /* Strategy 3: POWER_EFFICIENT (minimize energy) */
    printf("Policy: POWER_EFFICIENT\n");
    fb_set_dispatch_policy(FB_POLICY_POWER_EFFICIENT);
    fb_sgemm_auto('N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, m);
    fb_get_last_selection_explanation(&explanation);
    printf("  → Selected: %d (%s)\n\n",
           explanation.selected_device, explanation.reason);
    
    /* ========================================================================
     * 6. DATA LOCALITY - Demonstrate locality-aware dispatch
     * ====================================================================== */
    
    printf("=== DATA LOCALITY EXAMPLE ===\n");
    
    /* Register that data is on device 0 */
    if (device_count > 0) {
        fb_register_data_location(A, 0, m * k * sizeof(float));
        fb_register_data_location(B, 0, k * n * sizeof(float));
        
        printf("Data registered on device 0\n");
        printf("Using DATA_LOCALITY policy...\n");
        fb_set_dispatch_policy(FB_POLICY_DATA_LOCALITY);
        fb_sgemm_auto('N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, m);
        fb_get_last_selection_explanation(&explanation);
        printf("  → Selected: %d (should prefer device 0)\n\n",
               explanation.selected_device);
        
        fb_unregister_data_location(A);
        fb_unregister_data_location(B);
    }
    
    /* ========================================================================
     * 7. WORKLOAD-AWARE - Small vs Large problems
     * ====================================================================== */
    
    printf("=== WORKLOAD-AWARE DISPATCH ===\n");
    fb_set_dispatch_policy(FB_POLICY_ADAPTIVE);
    
    /* Small problem - likely picks CPU (avoids GPU kernel launch overhead) */
    int small_n = 64;
    printf("Small problem (%dx%d):\n", small_n, small_n);
    fb_sgemm_auto('N', 'N', small_n, small_n, small_n,
                 1.0f, A, small_n, B, small_n, 0.0f, C, small_n);
    fb_get_last_selection_explanation(&explanation);
    printf("  → Selected: %s\n",
           fb_device_get_type_string(fb_get_device_info(explanation.selected_device)));
    
    /* Large problem - likely picks GPU (parallelism wins) */
    int large_n = 4096;
    printf("Large problem (%dx%d):\n", large_n, large_n);
    fb_sgemm_auto('N', 'N', large_n, large_n, large_n,
                 1.0f, A, large_n, B, large_n, 0.0f, C, large_n);
    fb_get_last_selection_explanation(&explanation);
    printf("  → Selected: %s\n\n",
           fb_device_get_type_string(fb_get_device_info(explanation.selected_device)));
    
    /* ========================================================================
     * 8. STATISTICS
     * ====================================================================== */
    
    printf("=== LIBRARY STATISTICS ===\n");
    fb_print_status();
    
    /* ========================================================================
     * 9. Cleanup
     * ====================================================================== */
    
    free(A);
    free(B);
    free(C);
    
    fb_shutdown();
    
    printf("\n=== Example Complete ===\n");
    return 0;
}
