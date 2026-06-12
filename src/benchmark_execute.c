#include "benchmark_system.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#endif

// Constants from benchmark_cache.c
#define MAX_OPERATIONS      1248
#define MAX_BACKENDS        64
#define SIZE_CLASSES        5
#define SHAPE_CLASSES       5

// TODO: These will be replaced with actual backend operation calls
// For now, stubs to demonstrate the framework
extern int fb_execute_operation(uint32_t op_id, uint32_t backend_id, 
                                size_t m, size_t n, size_t k,
                                void* input, void* output);
extern bool fb_validate_output(void* output, size_t size, 
                               uint8_t* accuracy_mean, uint8_t* accuracy_worst);

// Timing helper
static unsigned int get_time_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (unsigned int)((count.QuadPart * 1000000000) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned int)ts.tv_sec * 1000000000 + (unsigned int)ts.tv_nsec;
#endif
}

// Calculate mean and standard deviation
static void calc_stats(unsigned int* samples, size_t count,
                      uint32_t* mean_out, uint32_t* stddev_out, uint32_t* p99_out) {
    if (count == 0) {
        *mean_out = *stddev_out = *p99_out = 0;
        return;
    }
    
    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += (double)samples[i];
    }
    double mean = sum / (double)count;
    *mean_out = (uint32_t)mean;
    
    // Standard deviation
    double variance = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = (double)samples[i] - mean;
        variance += diff * diff;
    }
    variance /= (double)count;
    *stddev_out = (uint32_t)sqrt(variance);
    
    // P99 (sort and pick 99th percentile)
    // Simple bubble sort for small arrays
    unsigned int* sorted = (unsigned int*)malloc(count * sizeof(unsigned int));
    memcpy(sorted, samples, count * sizeof(unsigned int));
    
    for (size_t i = 0; i < count - 1; i++) {
        for (size_t j = 0; j < count - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                unsigned int temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    
    size_t p99_idx = (count * 99) / 100;
    if (p99_idx >= count) p99_idx = count - 1;
    *p99_out = (uint32_t)sorted[p99_idx];
    
    free(sorted);
}

#ifdef _WIN32
// Windows: Use SEH (Structured Exception Handling)
static int run_protected_windows(uint32_t op_id, uint32_t backend_id,
                                 size_t m, size_t n, size_t k,
                                 void* input, void* output,
                                 uint32_t timeout_ms) {
    __try {
        // TODO: Actual operation execution
        // For now, return success
        return 0;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        // Caught exception (crash)
        return -1;
    }
}
#else
// Unix: Use fork + signal handling
static volatile sig_atomic_t alarm_triggered = 0;

static void alarm_handler(int sig) {
    (void)sig;
    alarm_triggered = 1;
}

static int run_protected_unix(uint32_t op_id, uint32_t backend_id,
                              size_t m, size_t n, size_t k,
                              void* input, void* output,
                              uint32_t timeout_ms) {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        // Set up alarm for timeout
        signal(SIGALRM, alarm_handler);
        alarm((timeout_ms + 999) / 1000); // Convert to seconds, round up
        
        // TODO: Execute operation
        // For now, exit success
        exit(0);
    } else if (pid > 0) {
        // Parent process
        int status;
        pid_t result = waitpid(pid, &status, 0);
        
        if (result == -1) {
            return -1; // Wait failed
        }
        
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            return -1; // Process crashed
        }
    }
    
    return -1; // Fork failed
}
#endif

int fb_benchmark_run_safe(uint32_t operation_id,
                          uint32_t backend_device_id,
                          const fb_benchmark_config_t* config,
                          fb_benchmark_result_t* result_out) {
    if (!config || !result_out) {
        return -1;
    }
    
    // Initialize result
    memset(result_out, 0, sizeof(fb_benchmark_result_t));
    result_out->success = false;
    
    // Allocate input/output buffers
    // TODO: Size based on operation type
    size_t buffer_size = config->m * config->n * sizeof(double);
    void* input = malloc(buffer_size);
    void* output = malloc(buffer_size);
    
    if (!input || !output) {
        free(input);
        free(output);
        result_out->error_message = "Memory allocation failed";
        return -1;
    }
    
    // Initialize input with test data
    memset(input, 0, buffer_size);
    
    // Timing samples
    unsigned int* time_samples = (unsigned int*)malloc(config->sample_iters * sizeof(unsigned int));
    if (!time_samples) {
        free(input);
        free(output);
        result_out->error_message = "Memory allocation failed";
        return -1;
    }
    
    size_t successful_samples = 0;
    
    // Warmup iterations
    for (uint32_t i = 0; i < config->warmup_iters; i++) {
#ifdef _WIN32
        int ret = run_protected_windows(operation_id, backend_device_id,
                                       config->m, config->n, config->k,
                                       input, output, config->timeout_ms);
#else
        int ret = run_protected_unix(operation_id, backend_device_id,
                                    config->m, config->n, config->k,
                                    input, output, config->timeout_ms);
#endif
        if (ret != 0) {
            // Warmup failed - operation is broken
            result_out->stats.flags = FB_BENCH_FLAG_CRASHED;
            result_out->error_message = "Operation crashed during warmup";
            free(time_samples);
            free(input);
            free(output);
            return 0; // Not an error - we detected the failure
        }
    }
    
    // Sample iterations
    for (uint32_t i = 0; i < config->sample_iters; i++) {
        unsigned int start = get_time_ns();
        
#ifdef _WIN32
        int ret = run_protected_windows(operation_id, backend_device_id,
                                       config->m, config->n, config->k,
                                       input, output, config->timeout_ms);
#else
        int ret = run_protected_unix(operation_id, backend_device_id,
                                    config->m, config->n, config->k,
                                    input, output, config->timeout_ms);
#endif
        
        unsigned int end = get_time_ns();
        
        if (ret != 0) {
            result_out->stats.flags = FB_BENCH_FLAG_CRASHED;
            result_out->error_message = "Operation crashed";
            break;
        }
        
        time_samples[successful_samples++] = end - start;
    }
    
    // Check if we got enough samples
    if (successful_samples == 0) {
        result_out->stats.flags |= FB_BENCH_FLAG_CRASHED;
        free(time_samples);
        free(input);
        free(output);
        return 0;
    }
    
    // Calculate timing statistics
    calc_stats(time_samples, successful_samples,
              &result_out->stats.time_mean_ns,
              &result_out->stats.time_stddev_ns,
              &result_out->stats.time_p99_ns);
    
    // Validate output if requested
    if (config->validate_output) {
        uint8_t accuracy_mean, accuracy_worst;
        // TODO: Actual validation against reference
        // For now, assume perfect accuracy
        accuracy_mean = accuracy_worst = 255;
        result_out->stats.accuracy_mean = accuracy_mean;
        result_out->stats.accuracy_worst = accuracy_worst;
        result_out->stats.precision_mean = 255;
        result_out->stats.precision_worst = 255;
    }
    
    result_out->stats.sample_count = (uint16_t)successful_samples;
    result_out->stats.flags |= FB_BENCH_FLAG_VALID;
    result_out->success = true;
    
    free(time_samples);
    free(input);
    free(output);
    
    return 0;
}

int fb_benchmark_run_full(uint32_t backend_device_id,
                          void* cache,
                          void (*progress_callback)(float progress)) {
    if (!cache) {
        return -1;
    }
    
    int successful = 0;
    int total_configs = MAX_OPERATIONS * SIZE_CLASSES * SHAPE_CLASSES;
    int processed = 0;
    
    // Iterate over all operations
    for (uint32_t op_id = 0; op_id < MAX_OPERATIONS; op_id++) {
        // Iterate over size classes
        for (fb_size_class_t size_class = 0; size_class < SIZE_CLASSES; size_class++) {
            // Iterate over shape classes
            for (fb_shape_class_t shape_class = 0; shape_class < SHAPE_CLASSES; shape_class++) {
                // Get representative dimensions
                size_t m, n, k;
                fb_get_representative_size(size_class, &m, &n, &k);
                
                size_t shape_m, shape_n;
                fb_get_representative_shape(shape_class, m, &shape_m, &shape_n);
                
                // Configure benchmark
                fb_benchmark_config_t config = {
                    .m = shape_m,
                    .n = shape_n,
                    .k = k,
                    .size_class = size_class,
                    .shape_class = shape_class,
                    .warmup_iters = 3,
                    .sample_iters = 10,
                    .timeout_ms = 5000,
                    .validate_output = true,
                    .detect_corruption = true
                };
                
                // Run benchmark
                fb_benchmark_result_t result;
                if (fb_benchmark_run_safe(op_id, backend_device_id, &config, &result) == 0) {
                    // Store result
                    fb_benchmark_cache_store(cache, op_id, backend_device_id,
                                           size_class, shape_class,
                                           &result.stats);
                    if (result.success) {
                        successful++;
                    }
                }
                
                processed++;
                if (progress_callback && (processed % 100 == 0)) {
                    progress_callback((float)processed / (float)total_configs);
                }
            }
        }
    }
    
    if (progress_callback) {
        progress_callback(1.0f);
    }
    
    return successful;
}

int fb_benchmark_run_quick(uint32_t backend_device_id,
                           void* cache,
                           void (*progress_callback)(float progress)) {
    // Quick mode: Only test representative sizes/shapes
    // Test each operation at MEDIUM size, SQUARE shape only
    
    if (!cache) {
        return -1;
    }
    
    int successful = 0;
    
    for (uint32_t op_id = 0; op_id < MAX_OPERATIONS; op_id++) {
        size_t m, n, k;
        fb_get_representative_size(FB_SIZE_MEDIUM, &m, &n, &k);
        
        fb_benchmark_config_t config = {
            .m = m,
            .n = n,
            .k = k,
            .size_class = FB_SIZE_MEDIUM,
            .shape_class = FB_SHAPE_SQUARE,
            .warmup_iters = 3,
            .sample_iters = 10,
            .timeout_ms = 5000,
            .validate_output = true,
            .detect_corruption = true
        };
        
        fb_benchmark_result_t result;
        if (fb_benchmark_run_safe(op_id, backend_device_id, &config, &result) == 0) {
            fb_benchmark_cache_store(cache, op_id, backend_device_id,
                                   FB_SIZE_MEDIUM, FB_SHAPE_SQUARE,
                                   &result.stats);
            if (result.success) {
                successful++;
            }
        }
        
        if (progress_callback && (op_id % 50 == 0)) {
            progress_callback((float)op_id / (float)MAX_OPERATIONS);
        }
    }
    
    if (progress_callback) {
        progress_callback(1.0f);
    }
    
    return successful;
}
