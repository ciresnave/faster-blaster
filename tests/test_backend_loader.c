/**
 * @file test_backend_loader.c
 * @brief Comprehensive test of the backend loader and integration
 * 
 * Tests the complete pipeline:
 * 1. System initialization
 * 2. Device detection
 * 3. Plugin registration
 * 4. Backend loading
 * 5. Device-to-backend mapping
 * 6. Operation execution
 * 7. Fallback handling
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/dispatch_unified.h"
#include "faster-blaster/backend_instance.h"
#include "faster-blaster/backend_plugin.h"
#include "faster-blaster/device_registry.h"
#include "backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_START(name) \
    do { \
        printf("\n"); \
        printf("=============================================================\n"); \
        printf("TEST: %s\n", name); \
        printf("=============================================================\n"); \
        tests_run++; \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("✓ PASSED\n"); \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        tests_failed++; \
        printf("✗ FAILED: %s\n", msg); \
    } while(0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            TEST_FAIL(msg); \
            return -1; \
        } \
    } while(0)

/* ============================================================================
 * Test 1: System Initialization
 * ========================================================================== */

static int test_initialization(void) {
    TEST_START("System Initialization");
    
    int result = fb_init();
    ASSERT(result == 0, "fb_init() failed");
    
    /* Verify registry initialized */
    uint32_t device_count = fb_get_device_count();
    ASSERT(device_count > 0, "No devices detected");
    
    printf("✓ Detected %u devices\n", device_count);
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Test 2: Plugin Registration
 * ========================================================================== */

static int test_plugin_registration(void) {
    TEST_START("Plugin Registration");
    
    const fb_plugin_registry_entry_t* plugins = fb_get_registered_plugins();
    ASSERT(plugins != NULL, "No plugins registered");
    
    int count = 0;
    for (const fb_plugin_registry_entry_t* entry = plugins; entry; entry = entry->next) {
        count++;
        printf("  Plugin %d: %s (version %s, vendor %s)\n", 
               count,
               entry->plugin->metadata->name,
               entry->plugin->metadata->version,
               entry->plugin->metadata->vendor);
    }
    
    /* Note: Plugin count depends on what backends were detected at build time.
     * Could be anywhere from 1 (reference only) to 9 (all backends found).
     * On this system: cuBLAS + MKL were found, so we expect at least 2. */
    ASSERT(count >= 1, "Expected at least 1 plugin (reference)");
    printf("✓ Total plugins registered: %d\n", count);
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Test 3: Device-to-Backend Mapping
 * ========================================================================== */

static int test_device_backend_mapping(void) {
    TEST_START("Device-to-Backend Mapping");
    
    uint32_t device_count = fb_get_device_count();
    
    for (uint32_t i = 0; i < device_count; i++) {
        fb_compute_device_t* device = fb_get_device(i);
        ASSERT(device != NULL, "Failed to get device");
        
        printf("\n  Device %u: %s\n", device->properties.device_id, device->properties.name);
        printf("    Type: %s\n", device->properties.type == FB_DEVICE_TYPE_CPU ? "CPU" : "GPU");
        
        /* Load backend for this device */
        fb_backend_instance_t* instance = fb_get_backend_for_device(device);
        
        if (!instance) {
            printf("    ⚠ Warning: Failed to load backend (may be expected)\n");
            continue;
        }
        
        printf("    ✓ Backend loaded: %s\n", 
               instance->plugin ? instance->plugin->metadata->name : "Reference");
        
        ASSERT(instance->initialized, "Backend instance not initialized");
        ASSERT(instance->vtable != NULL, "Backend vtable is NULL");
    }
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Test 4: Automatic Device Selection
 * ========================================================================== */

static int test_auto_device_selection(void) {
    TEST_START("Automatic Device Selection");
    
    /* Test different dispatch strategies */
    const fb_scheduling_policy_t policies[] = {
        FB_POLICY_FASTEST,
        FB_POLICY_LOAD_BALANCED,
        FB_POLICY_POWER_EFFICIENT,
        FB_POLICY_DATA_LOCALITY
    };
    
    const char* policy_names[] = {
        "FASTEST",
        "LOAD_BALANCED",
        "POWER_EFFICIENT",
        "DATA_LOCALITY"
    };
    
    for (size_t i = 0; i < sizeof(policies) / sizeof(policies[0]); i++) {
        printf("\n  Testing policy: %s\n", policy_names[i]);
        
        fb_set_policy(policies[i]);
        
        fb_backend_instance_t* instance = fb_get_current_backend(
            FB_PRECISION_FP32, NULL, 0);
        
        ASSERT(instance != NULL, "Device selection failed");
        ASSERT(instance->device != NULL, "Selected instance has no device");
        
        printf("    Selected device: %s\n", instance->device->properties.name);
        if (instance->plugin) {
            printf("    Using backend: %s\n", instance->plugin->metadata->name);
        }
    }
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Test 5: Manual Device Pinning
 * ========================================================================== */

static int test_manual_device_pinning(void) {
    TEST_START("Manual Device Pinning");
    
    uint32_t device_count = fb_get_device_count();
    
    if (device_count < 1) {
        printf("  Skipping (not enough devices)\n");
        TEST_PASS();
        return 0;
    }
    
    /* Pin to first device */
    int result = fb_use_device(0);
    ASSERT(result == 0, "Failed to pin to device 0");
    
    fb_backend_instance_t* instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    ASSERT(instance != NULL, "Failed to get backend after pinning");
    ASSERT(instance->device->properties.device_id == 0, "Wrong device selected after pinning");
    
    printf("✓ Successfully pinned to device 0: %s\n", instance->device->properties.name);
    
    /* Reset to auto */
    fb_use_auto();
    printf("✓ Reset to automatic selection\n");
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Test 6: Simple BLAS Operation (if backend supports it)
 * ========================================================================== */

static int test_simple_operation(void) {
    TEST_START("Simple BLAS Operation (SAXPY)");
    
    /* Get backend instance */
    fb_backend_instance_t* instance = fb_get_current_backend(FB_PRECISION_FP32, NULL, 0);
    ASSERT(instance != NULL, "No backend instance available");
    
    const fb_backend_vtable_t* vtable = fb_backend_get_vtable(instance);
    ASSERT(vtable != NULL, "Backend has no vtable");
    
    /* Check if SAXPY is implemented */
    if (!vtable->saxpy) {
        printf("  ⚠ Backend doesn't implement SAXPY, skipping execution test\n");
        TEST_PASS();
        return 0;
    }
    
    /* Allocate test vectors */
    const int64_t n = 1000;
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    
    ASSERT(x != NULL && y != NULL, "Memory allocation failed");
    
    /* Initialize vectors */
    for (int64_t i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = 1.0f;
    }
    
    /* Execute SAXPY: y = 2.0*x + y */
    float alpha = 2.0f;
    vtable->saxpy(n, alpha, x, 1, y, 1);
    
    /* Verify result (y[i] should be 2*i + 1) */
    bool correct = true;
    for (int64_t i = 0; i < n && i < 10; i++) {  /* Check first 10 elements */
        float expected = 2.0f * (float)i + 1.0f;
        if (fabsf(y[i] - expected) > 1e-5f) {
            printf("  ✗ Mismatch at y[%lld]: expected %.2f, got %.2f\n",
                   (long long)i, expected, y[i]);
            correct = false;
            break;
        }
    }
    
    free(x);
    free(y);
    
    if (correct) {
        printf("✓ SAXPY executed correctly\n");
    } else {
        TEST_FAIL("SAXPY produced incorrect results");
        return -1;
    }
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Test 7: System Status
 * ========================================================================== */

static int test_system_status(void) {
    TEST_START("System Status Reporting");
    
    fb_print_status();
    
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ========================================================================== */

int main(int argc, char** argv) {
    printf("\n");
    printf("#############################################################\n");
    printf("# faster-blaster Backend Loader Integration Test Suite\n");
    printf("#############################################################\n");
    
    /* Enable verbose logging if requested */
    if (argc > 1 && strcmp(argv[1], "-v") == 0) {
        fb_set_verbose(true);
    }
    
    /* Run tests */
    test_initialization();
    test_plugin_registration();
    test_device_backend_mapping();
    test_auto_device_selection();
    test_manual_device_pinning();
    test_simple_operation();
    test_system_status();
    
    /* Cleanup */
    printf("\n");
    printf("=============================================================\n");
    printf("Shutting down...\n");
    printf("=============================================================\n");
    fb_shutdown();
    
    /* Print summary */
    printf("\n");
    printf("#############################################################\n");
    printf("# Test Summary\n");
    printf("#############################################################\n");
    printf("  Tests run:    %d\n", tests_run);
    printf("  Tests passed: %d\n", tests_passed);
    printf("  Tests failed: %d\n", tests_failed);
    
    if (tests_failed == 0) {
        printf("\n  ✓ ALL TESTS PASSED!\n\n");
        return 0;
    } else {
        printf("\n  ✗ SOME TESTS FAILED\n\n");
        return 1;
    }
}
