/**
 * @file compute_manager.c
 * @brief Compute Manager Implementation
 * 
 * Central orchestrator for device selection and workload dispatch
 */

#include "faster-blaster/compute_manager.h"
#include "faster-blaster/device_registry.h"
#include "core/data_tracker.h"
#include "core/power_manager.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

// Global compute manager
static fb_compute_manager_t g_manager = {0};
static bool g_manager_initialized = false;

// ============================================================================
// Initialization
// ============================================================================

int fb_compute_manager_init(void) {
    if (g_manager_initialized) {
        return 0;
    }
    
    memset(&g_manager, 0, sizeof(g_manager));
    
    // Initialize subsystems
    fb_registry_init(FB_DISCOVERY_ALL);
    fb_data_tracker_init();
    fb_power_manager_init();
    
    // Default configuration
    g_manager.default_strategy = FB_STRATEGY_FASTEST;
    g_manager.enable_load_balancing = true;
    g_manager.enable_power_awareness = true;
    g_manager.enable_data_locality = true;
    
    // Default scoring weights
    fb_scoring_weights_t* weights = &g_manager.scoring_weights;
    weights->speed_weight = 0.4f;
    weights->load_weight = 0.2f;
    weights->locality_weight = 0.2f;
    weights->power_weight = 0.1f;
    weights->thermal_weight = 0.1f;
    
    g_manager_initialized = true;
    
    return 0;
}

void fb_compute_manager_shutdown(void) {
    if (!g_manager_initialized) return;
    
    fb_power_manager_shutdown();
    fb_data_tracker_shutdown();
    fb_registry_shutdown();
    
    memset(&g_manager, 0, sizeof(g_manager));
    g_manager_initialized = false;
}

// ============================================================================
// Configuration
// ============================================================================

void fb_set_dispatch_strategy(fb_dispatch_strategy_t strategy) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    g_manager.default_strategy = strategy;
}

fb_dispatch_strategy_t fb_get_dispatch_strategy(void) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    return g_manager.default_strategy;
}

void fb_set_scoring_weights(const fb_scoring_weights_t* weights) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    g_manager.scoring_weights = *weights;
    
    // Normalize weights to sum to 1.0
    float total = weights->speed_weight + weights->load_weight + 
                  weights->locality_weight + weights->power_weight + 
                  weights->thermal_weight;
    
    if (total > 0.0f) {
        g_manager.scoring_weights.speed_weight /= total;
        g_manager.scoring_weights.load_weight /= total;
        g_manager.scoring_weights.locality_weight /= total;
        g_manager.scoring_weights.power_weight /= total;
        g_manager.scoring_weights.thermal_weight /= total;
    }
}

void fb_get_scoring_weights(fb_scoring_weights_t* weights) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    *weights = g_manager.scoring_weights;
}

// ============================================================================
// Multi-Factor Scoring Algorithm
// ============================================================================

static float calculate_speed_score(fb_compute_device_t* device, 
                                   fb_precision_t precision) {
    // Higher GFLOPS = higher score (0.0 to 1.0)
    double gflops = fb_device_get_gflops(device, precision);
    
    // Normalize against maximum possible GFLOPS
    // (RTX 4090 ~82 TFLOPS FP32 as reference)
    double max_gflops = 82000.0;
    float score = (float)(gflops / max_gflops);
    
    // Clamp to [0, 1]
    if (score > 1.0f) score = 1.0f;
    
    return score;
}

static float calculate_load_score(fb_compute_device_t* device) {
    // Lower load = higher score
    float load = device->current_load;
    
    // Invert: 0% load = 1.0 score, 100% load = 0.0 score
    float score = 1.0f - load;
    
    // Clamp
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;
    
    return score;
}

static float calculate_locality_score(fb_compute_device_t* device,
                                      const void** data_ptrs,
                                      uint32_t num_data_ptrs) {
    if (!data_ptrs || num_data_ptrs == 0) {
        return 0.5f; // Neutral if no data specified
    }
    
    uint32_t data_on_device = 0;
    
    for (uint32_t i = 0; i < num_data_ptrs; i++) {
        if (fb_data_is_on_device(data_ptrs[i], device)) {
            data_on_device++;
        }
    }
    
    // Fraction of data already on device
    return (float)data_on_device / (float)num_data_ptrs;
}

static float calculate_power_score(fb_compute_device_t* device) {
    // Lower power consumption = higher score (when power-aware)
    
    uint32_t power = device->power.tdp_watts;
    if (power == 0) {
        // Estimate: CPU ~65W, GPU ~200W
        power = (device->type == FB_DEVICE_GPU) ? 200 : 65;
    }
    
    // Normalize against typical max (RTX 4090 TDP = 450W)
    float normalized = (float)power / 450.0f;
    
    // Invert: lower power = higher score
    float score = 1.0f - normalized;
    
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;
    
    return score;
}

static float calculate_thermal_score(fb_compute_device_t* device) {
    // Lower temperature = higher score
    
    uint32_t temp = device->thermal.current_temp_c;
    uint32_t max_temp = device->thermal.max_temp_c;
    
    if (max_temp == 0) max_temp = 90; // Default safe max
    if (temp == 0) return 1.0f; // Unknown temp = assume cool
    
    // Normalize temperature
    float temp_ratio = (float)temp / (float)max_temp;
    
    // Invert: cooler = higher score
    float score = 1.0f - temp_ratio;
    
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;
    
    return score;
}

static float calculate_total_score(fb_compute_device_t* device,
                                   fb_precision_t precision,
                                   const void** data_ptrs,
                                   uint32_t num_data_ptrs,
                                   const fb_scoring_weights_t* weights) {
    float speed_score = calculate_speed_score(device, precision);
    float load_score = calculate_load_score(device);
    float locality_score = calculate_locality_score(device, data_ptrs, num_data_ptrs);
    float power_score = calculate_power_score(device);
    float thermal_score = calculate_thermal_score(device);
    
    float total = speed_score * weights->speed_weight +
                  load_score * weights->load_weight +
                  locality_score * weights->locality_weight +
                  power_score * weights->power_weight +
                  thermal_score * weights->thermal_weight;
    
    return total;
}

// ============================================================================
// Device Selection
// ============================================================================

fb_compute_device_t* fb_select_device(fb_dispatch_strategy_t strategy,
                                     fb_precision_t precision,
                                     const void** data_ptrs,
                                     uint32_t num_data_ptrs) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    // Update device states
    fb_update_all_devices();
    fb_power_manager_update();
    
    fb_device_registry_t* registry = fb_get_registry();
    
    switch (strategy) {
        case FB_STRATEGY_FASTEST: {
            // Pure speed - find fastest device for this precision
            fb_compute_device_t* fastest = NULL;
            double max_gflops = 0.0;
            
            for (uint32_t i = 0; i < registry->num_devices; i++) {
                fb_compute_device_t* dev = &registry->devices[i];
                if (!fb_is_device_available(dev)) continue;
                if (!fb_device_supports_precision(dev, precision)) continue;
                
                double gflops = fb_device_get_gflops(dev, precision);
                if (gflops > max_gflops) {
                    max_gflops = gflops;
                    fastest = dev;
                }
            }
            return fastest;
        }
        
        case FB_STRATEGY_LOAD_BALANCED: {
            // Find least loaded device
            return fb_find_least_loaded_device();
        }
        
        case FB_STRATEGY_POWER_EFFICIENT: {
            // Prefer CPU if on battery, otherwise use scoring
            if (fb_should_prefer_cpu()) {
                return fb_get_cpu_device();
            }
            
            // Use scoring with high power weight
            fb_scoring_weights_t weights = g_manager.scoring_weights;
            weights.power_weight = 0.5f;
            weights.speed_weight = 0.3f;
            weights.load_weight = 0.1f;
            weights.locality_weight = 0.05f;
            weights.thermal_weight = 0.05f;
            
            return fb_select_device_scored(precision, data_ptrs, num_data_ptrs, &weights);
        }
        
        case FB_STRATEGY_DATA_LOCALITY: {
            // Prefer device with most data already present
            if (!data_ptrs || num_data_ptrs == 0) {
                // No data specified, fall back to fastest
                return fb_find_fastest_device(precision);
            }
            
            fb_compute_device_t* best = NULL;
            uint32_t max_local_data = 0;
            
            for (uint32_t i = 0; i < registry->num_devices; i++) {
                fb_compute_device_t* dev = &registry->devices[i];
                if (!fb_is_device_available(dev)) continue;
                if (!fb_device_supports_precision(dev, precision)) continue;
                
                uint32_t local_data = 0;
                for (uint32_t j = 0; j < num_data_ptrs; j++) {
                    if (fb_data_is_on_device(data_ptrs[j], dev)) {
                        local_data++;
                    }
                }
                
                if (local_data > max_local_data) {
                    max_local_data = local_data;
                    best = dev;
                }
            }
            
            return best ? best : fb_find_fastest_device(precision);
        }
        
        case FB_STRATEGY_ADAPTIVE: {
            // Use full multi-factor scoring
            return fb_select_device_scored(precision, data_ptrs, num_data_ptrs,
                                          &g_manager.scoring_weights);
        }
        
        case FB_STRATEGY_CUSTOM: {
            // Use custom scoring weights if set
            return fb_select_device_scored(precision, data_ptrs, num_data_ptrs,
                                          &g_manager.scoring_weights);
        }
        
        default:
            return fb_find_fastest_device(precision);
    }
}

fb_compute_device_t* fb_select_device_scored(fb_precision_t precision,
                                            const void** data_ptrs,
                                            uint32_t num_data_ptrs,
                                            const fb_scoring_weights_t* weights) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    fb_device_registry_t* registry = fb_get_registry();
    
    fb_compute_device_t* best = NULL;
    float best_score = -1.0f;
    
    for (uint32_t i = 0; i < registry->num_devices; i++) {
        fb_compute_device_t* dev = &registry->devices[i];
        
        if (!fb_is_device_available(dev)) continue;
        if (!fb_device_supports_precision(dev, precision)) continue;
        
        float score = calculate_total_score(dev, precision, data_ptrs, 
                                           num_data_ptrs, weights);
        
        if (score > best_score) {
            best_score = score;
            best = dev;
        }
    }
    
    return best;
}

fb_compute_device_t* fb_select_device_auto(void) {
    if (!g_manager_initialized) {
        fb_compute_manager_init();
    }
    
    return fb_select_device(g_manager.default_strategy, FB_PRECISION_FP32, 
                          NULL, 0);
}

// ============================================================================
// Load Estimation
// ============================================================================

double fb_estimate_operation_cost(fb_blas_op_t operation,
                                  size_t m, size_t n, size_t k,
                                  fb_precision_t precision) {
    // Estimate FLOPs for the operation
    uint flops = 0;
    
    switch (operation) {
        // Level 1 (Vector-Vector)
        case FB_OP_AXPY:
        case FB_OP_SCAL:
        case FB_OP_COPY:
            flops = m; // O(n)
            break;
            
        case FB_OP_DOT:
        case FB_OP_NRM2:
        case FB_OP_ASUM:
            flops = 2 * m; // O(2n) for multiply-add
            break;
            
        // Level 2 (Matrix-Vector)
        case FB_OP_GEMV:
        case FB_OP_SYMV:
            flops = 2 * m * n; // O(2mn)
            break;
            
        case FB_OP_GER:
            flops = 2 * m * n; // O(2mn)
            break;
            
        // Level 3 (Matrix-Matrix)
        case FB_OP_GEMM:
        case FB_OP_SYMM:
            flops = 2 * m * n * k; // O(2mnk)
            break;
            
        case FB_OP_TRSM:
        case FB_OP_TRMM:
            flops = m * n * k; // O(mnk)
            break;
            
        default:
            flops = m * n; // Default estimate
            break;
    }
    
    // Convert to GFLOPs
    return (double)flops / 1e9;
}

double fb_estimate_execution_time(fb_compute_device_t* device,
                                  double gflops_required) {
    if (!device || device->performance.fp32_gflops == 0) {
        return 1.0; // Default 1 second
    }
    
    // Time = GFLOPs / (Device GFLOPS * efficiency)
    double efficiency = 0.8; // Assume 80% of peak performance
    double device_gflops = device->performance.fp32_gflops;
    
    double time_s = gflops_required / (device_gflops * efficiency);
    
    return time_s;
}

// ============================================================================
// Statistics
// ============================================================================

void fb_record_execution(fb_compute_device_t* device, 
                        fb_blas_op_t operation,
                        double execution_time_s) {
    if (!device) return;
    
    // Update device load (simple moving average)
    float new_load = (execution_time_s > 0.1f) ? 0.8f : 0.2f;
    device->current_load = device->current_load * 0.9f + new_load * 0.1f;
    
    // TODO: Store in history for adaptive learning
}

void fb_print_manager_stats(void) {
    printf("=== Compute Manager Statistics ===\n");
    printf("Strategy: ");
    
    switch (g_manager.default_strategy) {
        case FB_STRATEGY_FASTEST: printf("FASTEST\n"); break;
        case FB_STRATEGY_LOAD_BALANCED: printf("LOAD_BALANCED\n"); break;
        case FB_STRATEGY_POWER_EFFICIENT: printf("POWER_EFFICIENT\n"); break;
        case FB_STRATEGY_DATA_LOCALITY: printf("DATA_LOCALITY\n"); break;
        case FB_STRATEGY_ADAPTIVE: printf("ADAPTIVE\n"); break;
        case FB_STRATEGY_CUSTOM: printf("CUSTOM\n"); break;
    }
    
    printf("\nScoring Weights:\n");
    printf("  Speed: %.2f\n", g_manager.scoring_weights.speed_weight);
    printf("  Load: %.2f\n", g_manager.scoring_weights.load_weight);
    printf("  Locality: %.2f\n", g_manager.scoring_weights.locality_weight);
    printf("  Power: %.2f\n", g_manager.scoring_weights.power_weight);
    printf("  Thermal: %.2f\n", g_manager.scoring_weights.thermal_weight);
    
    printf("\nFeatures:\n");
    printf("  Load Balancing: %s\n", g_manager.enable_load_balancing ? "Enabled" : "Disabled");
    printf("  Power Awareness: %s\n", g_manager.enable_power_awareness ? "Enabled" : "Disabled");
    printf("  Data Locality: %s\n", g_manager.enable_data_locality ? "Enabled" : "Disabled");
}
