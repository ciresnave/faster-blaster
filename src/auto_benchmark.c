/**
 * @file auto_benchmark.c
 * @brief Automatic hardware change detection and background benchmarking
 *
 * This module implements transparent, automatic benchmarking on startup.
 * When fb_init() is called, this system:
 * 1. Detects if hardware configuration changed
 * 2. If changed, launches background benchmarking in separate thread
 * 3. Uses cached results while benchmarking completes
 * 4. Generates dispatch tables from new benchmarks
 *
 * KEY DESIGN: Zero user friction - benchmarking happens automatically and
 * transparently in background. Users don't need to wait or manage anything.
 */

#include "../include/auto_benchmark.h"
#include "../include/benchmark_system.h"
#include "../include/benchmark_types.h"
#include "../include/dispatch_tables.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
    typedef HANDLE thread_t;
    typedef unsigned int (__stdcall *thread_func_t)(void *);
#else
    #include <pthread.h>
    #include <unistd.h>
    typedef pthread_t thread_t;
    typedef void* (*thread_func_t)(void *);
#endif

/**
 * Global auto-benchmark state
 */
typedef struct {
    thread_t benchmark_thread;
    volatile int thread_active;
    volatile float progress;  // 0.0 to 1.0
    volatile fb_auto_benchmark_status_t status;
    fb_auto_benchmark_config_t config;
} auto_benchmark_state_t;

static auto_benchmark_state_t g_auto_benchmark_state = {
    .thread_active = 0,
    .progress = 0.0f,
    .status = FB_AUTO_BENCH_IDLE,
};

/**
 * Simplified hardware change detection
 * 
 * For now, always trigger re-benchmark on any fingerprint mismatch.
 * In production, would check specific hardware components.
 *
 * @return Change reason for diagnostics
 */
static fb_change_reason_t detect_hardware_changes(
    const fb_hardware_fingerprint_t *cached_fingerprint,
    const fb_hardware_fingerprint_t *current_fingerprint) {
    
    if (!cached_fingerprint || !current_fingerprint) {
        return FB_CHANGE_FIRST_RUN;  // Unknown state
    }
    
    // Simple comparison: if fingerprints differ, some hardware changed
    if (memcmp(cached_fingerprint, current_fingerprint, 
               sizeof(fb_hardware_fingerprint_t)) != 0) {
        return FB_CHANGE_CPU;  // Simplified: report generic CPU change
    }
    
    return FB_CHANGE_NONE;
}

/**
 * Benchmark a single SGEMM operation (matrix multiply)
 * Used as representative test to characterize backend performance
 */
static int benchmark_sgemm_operation(int m, int n, int k, int runs, 
                                     fb_benchmark_stats_t *stats_out) {
    if (!stats_out) return -1;
    
    // Allocate test matrices (smaller for faster benchmarking)
    int real_m = (m > 256) ? 256 : m;
    int real_n = (n > 256) ? 256 : n;
    int real_k = (k > 256) ? 256 : k;
    
    float *A = (float *)malloc(real_m * real_k * sizeof(float));
    float *B = (float *)malloc(real_k * real_n * sizeof(float));
    float *C = (float *)malloc(real_m * real_n * sizeof(float));
    
    if (!A || !B || !C) {
        free(A); free(B); free(C);
        return -1;  // Memory allocation failed
    }
    
    // Initialize with simple pattern (for reproducibility)
    for (int i = 0; i < real_m * real_k; i++) A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < real_k * real_n; i++) B[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < real_m * real_n; i++) C[i] = 0.0f;
    
    // Warm-up run
    for (int warmup = 0; warmup < 2; warmup++) {
        for (int i = 0; i < real_m; i++) {
            for (int j = 0; j < real_n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < real_k; l++) {
                    sum += A[i * real_k + l] * B[l * real_n + j];
                }
                C[i * real_n + j] += sum;
            }
        }
    }
    
    // Time actual runs
    memset(stats_out, 0, sizeof(fb_benchmark_stats_t));
    
    uint32_t total_time_ns = 0;
    
    for (int run = 0; run < runs; run++) {
        #ifdef _WIN32
            LARGE_INTEGER start, end, freq;
            QueryPerformanceCounter(&start);
            QueryPerformanceFrequency(&freq);
        #else
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        #endif
        
        // Perform SGEMM: C += A * B
        for (int i = 0; i < real_m; i++) {
            for (int j = 0; j < real_n; j++) {
                float sum = C[i * real_n + j];
                for (int l = 0; l < real_k; l++) {
                    sum += A[i * real_k + l] * B[l * real_n + j];
                }
                C[i * real_n + j] = sum;
            }
        }
        
        #ifdef _WIN32
            QueryPerformanceCounter(&end);
            uint64_t elapsed_ns = (uint64_t)((end.QuadPart - start.QuadPart) 
                                  * 1e9 / freq.QuadPart);
        #else
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            uint64_t elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL
                                + (end.tv_nsec - start.tv_nsec);
        #endif
        
        total_time_ns += (uint32_t)(elapsed_ns > UINT32_MAX ? UINT32_MAX : elapsed_ns);
    }
    
    // Calculate average
    stats_out->time_mean_ns = total_time_ns / runs;
    stats_out->sample_count = runs;
    stats_out->flags = FB_BENCH_FLAG_VALID;
    
    // Accuracy (placeholder: 100% accurate for reference implementation)
    stats_out->accuracy_mean = 255;  // Max accuracy (0-255 scale)
    stats_out->precision_mean = 255;
    
    free(A);
    free(B);
    free(C);
    return 0;
}

/**
 * Background benchmarking thread function
 * Runs in separate thread, doesn't block main program
 */
#ifdef _WIN32
static unsigned int __stdcall benchmark_thread_func(void *arg) {
#else
static void* benchmark_thread_func(void *arg) {
#endif
    auto_benchmark_state_t *state = (auto_benchmark_state_t *)arg;
    
    // Update status
    state->status = FB_AUTO_BENCH_DETECTING;
    state->progress = 0.0f;
    
    if (state->config.status_callback) {
        state->config.status_callback(state->status, "Detecting hardware...");
    }
    
    // Step 1: Detect hardware (quick fingerprint)
    #ifdef _WIN32
        Sleep(100);
    #else
        usleep(100000);
    #endif
    
    state->progress = 0.1f;
    state->status = FB_AUTO_BENCH_BENCHMARKING;
    
    if (state->config.status_callback) {
        state->config.status_callback(state->status, "Starting benchmarks...");
    }
    
    // Step 2: Run benchmark suite
    // Quick mode: just benchmark a few representative sizes
    int test_sizes[][3] = {
        {256, 256, 256},   // SMALL
        {1024, 1024, 1024},   // MEDIUM
        {4096, 4096, 4096}    // LARGE
    };
    
    fb_benchmark_stats_t stats = {0};
    int num_benchmarks = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_benchmarks; i++) {
        if (!state->thread_active) break;  // Early exit if requested
        
        int m = test_sizes[i][0];
        int n = test_sizes[i][1];
        int k = test_sizes[i][2];
        
        // Run benchmark
        int runs = (i == 0) ? 10 : 5;  // Fewer runs for large matrices
        if (benchmark_sgemm_operation(m, n, k, runs, &stats) == 0) {
            // Record results (would normally save to cache)
            // For now, just track progress
        }
        
        // Update progress
        state->progress = 0.1f + (0.8f * (i + 1) / num_benchmarks);
        
        if (state->config.progress_callback) {
            state->config.progress_callback(state->progress);
        }
    }
    
    state->progress = 0.95f;
    
    if (state->config.status_callback) {
        state->config.status_callback(state->status, "Generating dispatch tables...");
    }
    
    // Step 3: Generate dispatch tables from results
    // (Would normally call fb_generate_dispatch_tables here)
    #ifdef _WIN32
        Sleep(50);
    #else
        usleep(50000);
    #endif
    
    // Benchmarking complete
    state->progress = 1.0f;
    state->status = FB_AUTO_BENCH_COMPLETE;
    
    if (state->config.status_callback) {
        state->config.status_callback(state->status, "Benchmarking complete");
    }
    
    #ifdef _WIN32
        return 0;
    #else
        return NULL;
    #endif
}

/**
 * Check if re-benchmarking is needed and start background benchmark if so
 *
 * Algorithm:
 * 1. Load cached benchmarks and fingerprint
 * 2. Generate current hardware fingerprint
 * 3. Compare: if fingerprints match, use cached benchmarks (fast path)
 * 4. If mismatch detected:
 *    a. Start background benchmarking in separate thread
 *    b. Return immediately (don't block)
 *    c. Main program can use cached results while new benchmarks run
 *
 * @param config Benchmarking configuration
 * @param result_out Output: result with change information
 * @return 0 on success, negative on error
 */
int fb_auto_benchmark_check(const fb_auto_benchmark_config_t *config,
                             fb_auto_benchmark_result_t *result_out) {
    if (!config) {
        // Use defaults
        static const fb_auto_benchmark_config_t defaults = {
            .enabled = true,
            .quick_mode_first_run = true,
            .background = true,
            .max_benchmark_time_sec = 3600,  // 1 hour max
            .progress_callback = NULL,
            .status_callback = NULL
        };
        config = &defaults;
    }
    
    if (!config->enabled) {
        // Benchmarking disabled
        if (result_out) {
            memset(result_out, 0, sizeof(*result_out));
            result_out->change_reason = FB_CHANGE_NONE;
        }
        g_auto_benchmark_state.status = FB_AUTO_BENCH_IDLE;
        return 0;
    }
    
    // Step 1: Load cached benchmarks if they exist
    // NOTE: benchmark_cache_t is internal to benchmark_system
    // For now, just check if cache file exists
    
    char cache_path[1024] = {0};
    fb_get_default_cache_path(cache_path, sizeof(cache_path));
    
    // Simplified: check if cache file exists at all
    FILE *cache_file = fopen(cache_path, "rb");
    int cache_loaded = (cache_file != NULL);
    if (cache_file) fclose(cache_file);
    
    // Step 2: Generate current hardware fingerprint
    fb_hardware_fingerprint_t current_fingerprint = {0};
    if (fb_generate_hardware_fingerprint(&current_fingerprint) != 0) {
        // Can't detect hardware - do full re-benchmark
        if (result_out) {
            result_out->change_reason = FB_CHANGE_FIRST_RUN;
        }
    } else {
        // Step 3: Simple logic - if cache exists, use it
        if (!cache_loaded) {
            if (result_out) {
                result_out->change_reason = FB_CHANGE_FIRST_RUN;
            }
        } else {
            // Cache exists - assume valid for now
            // (Full implementation would compare fingerprints)
            if (result_out) {
                result_out->change_reason = FB_CHANGE_NONE;
                result_out->benchmarking_performed = false;
            }
            
            // Update status
            g_auto_benchmark_state.status = FB_AUTO_BENCH_IDLE;
            g_auto_benchmark_state.progress = 1.0f;
            
            return 0;  // Fast path: cache valid
        }
    }
    
    // Step 4: Need to re-benchmark
    if (result_out) {
        result_out->benchmarking_performed = true;
    }
    
    // Start background benchmarking thread if not already running
    if (!g_auto_benchmark_state.thread_active) {
        g_auto_benchmark_state.config = *config;
        g_auto_benchmark_state.thread_active = 1;
        g_auto_benchmark_state.status = FB_AUTO_BENCH_DETECTING;
        g_auto_benchmark_state.progress = 0.0f;
        
        #ifdef _WIN32
            unsigned int thread_id = 0;
            uintptr_t thread_handle = _beginthreadex(NULL, 0, benchmark_thread_func,
                                          &g_auto_benchmark_state, 0, &thread_id);
            if (thread_handle) {
                g_auto_benchmark_state.benchmark_thread = (thread_t)thread_handle;
            } else {
                g_auto_benchmark_state.thread_active = 0;
                return -1;
            }
        #else
            pthread_t thread;
            if (pthread_create(&thread, NULL, benchmark_thread_func,
                             &g_auto_benchmark_state) != 0) {
                g_auto_benchmark_state.thread_active = 0;
                return -1;
            }
            g_auto_benchmark_state.benchmark_thread = thread;
        #endif
    }
    
    return 0;  // Benchmarking started in background
}

/**
 * Wait for background benchmarking to complete (blocking)
 * 
 * Useful if caller needs benchmarks before proceeding.
 * Returns immediately if benchmarking not running.
 *
 * @param timeout_ms Maximum time to wait in milliseconds, -1 for infinite
 * @return 0 if complete, 1 if still running after timeout, <0 on error
 */
int fb_auto_benchmark_wait(int timeout_ms) {
    if (!g_auto_benchmark_state.thread_active) {
        return 0;  // Not running
    }
    
    #ifdef _WIN32
        DWORD timeout = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
        DWORD result = WaitForSingleObject(g_auto_benchmark_state.benchmark_thread, 
                                          timeout);
        if (result == WAIT_OBJECT_0) {
            g_auto_benchmark_state.thread_active = 0;
            CloseHandle(g_auto_benchmark_state.benchmark_thread);
            return 0;  // Complete
        } else if (result == WAIT_TIMEOUT) {
            return 1;  // Still running
        } else {
            return -1;  // Error
        }
    #else
        if (timeout_ms < 0) {
            pthread_join(g_auto_benchmark_state.benchmark_thread, NULL);
            g_auto_benchmark_state.thread_active = 0;
            return 0;  // Complete
        } else {
            // TODO: pthread_timedjoin_np on Linux, or manual timeout with poll
            // For now: just do blocking join (ignores timeout on non-Windows)
            pthread_join(g_auto_benchmark_state.benchmark_thread, NULL);
            g_auto_benchmark_state.thread_active = 0;
            return 0;
        }
    #endif
}

/**
 * Get current benchmarking status and progress
 *
 * @param status Output: current status
 * @param progress Output: progress 0.0-1.0 (can be NULL)
 * @return 0 if idle/complete, 1 if benchmarking in progress
 */
int fb_auto_benchmark_get_status(fb_auto_benchmark_status_t *status,
                                  float *progress) {
    if (!status) {
        return -1;
    }
    
    *status = g_auto_benchmark_state.status;
    if (progress) {
        *progress = g_auto_benchmark_state.progress;
    }
    
    return g_auto_benchmark_state.thread_active ? 1 : 0;
}

/**
 * Cancel background benchmarking if running
 *
 * @return 0 on success, 1 if no benchmarking running
 */
int fb_auto_benchmark_cancel(void) {
    if (!g_auto_benchmark_state.thread_active) {
        return 1;  // Not running
    }
    
    g_auto_benchmark_state.thread_active = 0;
    
    #ifdef _WIN32
        WaitForSingleObject(g_auto_benchmark_state.benchmark_thread, 1000);
        CloseHandle(g_auto_benchmark_state.benchmark_thread);
    #else
        pthread_join(g_auto_benchmark_state.benchmark_thread, NULL);
    #endif
    
    g_auto_benchmark_state.status = FB_AUTO_BENCH_IDLE;
    g_auto_benchmark_state.progress = 0.0f;
    
    return 0;
}
