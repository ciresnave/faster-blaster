/**
 * @file operation_router.c
 * @brief Implementation of operation-level backend routing
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/operation_router.h"
#include "faster-blaster/device_registry.h"
#include "faster-blaster/backend_instance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default size thresholds for routing decisions */
#define DEFAULT_LEVEL1_GPU_THRESHOLD  10000    /* Use GPU for vectors > 10K elements */
#define DEFAULT_LEVEL2_GPU_THRESHOLD  1000     /* Use GPU for matrices > 1000 elements */
#define DEFAULT_LEVEL3_GPU_THRESHOLD  128      /* Use GPU for GEMM when M,N,K > 128 */

/* Global routing state */
static struct {
    bool initialized;
    fb_routing_policy_t policy;
    fb_route_config_t configs[FB_OP_CATEGORY_COUNT];
    fb_routing_callback_t custom_callback;
    void* callback_user_data;
} g_routing = {0};

int fb_routing_init(void) {
    if (g_routing.initialized) {
        return 0;
    }
    
    /* Set default policy */
    g_routing.policy = FB_ROUTE_AUTO;
    
    /* Initialize default configurations */
    for (int i = 0; i < FB_OP_CATEGORY_COUNT; i++) {
        g_routing.configs[i].preferred_backend = NULL;  /* Auto-select */
        g_routing.configs[i].device_id = -1;  /* Any device */
        g_routing.configs[i].size_threshold = 0;
    }
    
    /* Set reasonable defaults for common operations */
    /* Level 1: Small operations stay on CPU, large go to GPU */
    g_routing.configs[FB_OP_LEVEL1_DOT].size_threshold = DEFAULT_LEVEL1_GPU_THRESHOLD;
    g_routing.configs[FB_OP_LEVEL1_AXPY].size_threshold = DEFAULT_LEVEL1_GPU_THRESHOLD;
    
    /* Level 2: Medium threshold */
    g_routing.configs[FB_OP_LEVEL2_GEMV].size_threshold = DEFAULT_LEVEL2_GPU_THRESHOLD;
    g_routing.configs[FB_OP_LEVEL2_GER].size_threshold = DEFAULT_LEVEL2_GPU_THRESHOLD;
    g_routing.configs[FB_OP_LEVEL2_SYMV].size_threshold = DEFAULT_LEVEL2_GPU_THRESHOLD;
    
    /* Level 3: Low threshold - GPUs excel at large matrix operations */
    g_routing.configs[FB_OP_LEVEL3_GEMM].size_threshold = DEFAULT_LEVEL3_GPU_THRESHOLD * DEFAULT_LEVEL3_GPU_THRESHOLD;
    g_routing.configs[FB_OP_LEVEL3_SYMM].size_threshold = DEFAULT_LEVEL3_GPU_THRESHOLD * DEFAULT_LEVEL3_GPU_THRESHOLD;
    g_routing.configs[FB_OP_LEVEL3_TRSM].size_threshold = DEFAULT_LEVEL3_GPU_THRESHOLD * DEFAULT_LEVEL3_GPU_THRESHOLD;
    
    /* LAPACK: Typically benefit from GPU for larger problems */
    g_routing.configs[FB_OP_LAPACK_SOLVE].size_threshold = 256 * 256;
    g_routing.configs[FB_OP_LAPACK_FACTOR].size_threshold = 256 * 256;
    g_routing.configs[FB_OP_LAPACK_EIGEN].size_threshold = 512 * 512;
    
    g_routing.initialized = true;
    printf("[INFO] Operation routing initialized with AUTO policy\n");
    return 0;
}

int fb_routing_set_policy(fb_routing_policy_t policy) {
    if (!g_routing.initialized) {
        fprintf(stderr, "[ERROR] Routing system not initialized\n");
        return -1;
    }
    
    g_routing.policy = policy;
    printf("[INFO] Routing policy set to: %d\n", policy);
    return 0;
}

int fb_routing_configure(fb_operation_category_t category, const fb_route_config_t* config) {
    if (!g_routing.initialized) {
        fprintf(stderr, "[ERROR] Routing system not initialized\n");
        return -1;
    }
    
    if (category >= FB_OP_CATEGORY_COUNT) {
        fprintf(stderr, "[ERROR] Invalid operation category: %d\n", category);
        return -1;
    }
    
    if (!config) {
        fprintf(stderr, "[ERROR] NULL config provided\n");
        return -1;
    }
    
    /* Copy configuration */
    g_routing.configs[category] = *config;
    
    printf("[INFO] Configured routing for category %d: backend=%s, threshold=%zu, device=%d\n",
           category,
           config->preferred_backend ? config->preferred_backend : "auto",
           config->size_threshold,
           config->device_id);
    
    return 0;
}

int fb_routing_set_callback(fb_routing_callback_t callback, void* user_data) {
    if (!g_routing.initialized) {
        fprintf(stderr, "[ERROR] Routing system not initialized\n");
        return -1;
    }
    
    g_routing.custom_callback = callback;
    g_routing.callback_user_data = user_data;
    
    printf("[INFO] Custom routing callback set\n");
    return 0;
}

int fb_routing_get_device(fb_operation_category_t category, size_t problem_size) {
    if (!g_routing.initialized) {
        /* Return current device if routing not initialized */
        return -1;
    }
    
    /* Handle custom routing policy */
    if (g_routing.policy == FB_ROUTE_CUSTOM && g_routing.custom_callback) {
        return g_routing.custom_callback(category, problem_size, g_routing.callback_user_data);
    }
    
    /* Handle by-device policy (use current device) */
    if (g_routing.policy == FB_ROUTE_BY_DEVICE) {
        return -1;  /* Use current device */
    }
    
    /* Get configuration for this operation */
    const fb_route_config_t* config = &g_routing.configs[category];
    
    /* Check if specific device is requested */
    if (config->device_id >= 0) {
        return config->device_id;
    }
    
    /* AUTO or BY_SIZE or BY_OPERATION policy */
    /* Decide CPU vs GPU based on problem size */
    
    /* Get device counts */
    int cpu_count = fb_device_get_cpu_count();
    int gpu_count = fb_device_get_gpu_count();
    
    /* If no GPUs available, use CPU */
    if (gpu_count == 0) {
        return 0;  /* First CPU device */
    }
    
    /* Size-based routing: small problems on CPU, large on GPU */
    if (problem_size < config->size_threshold) {
        /* Use CPU for small problems */
        return 0;
    } else {
        /* Use first GPU for large problems */
        return cpu_count;  /* First GPU device ID */
    }
}

