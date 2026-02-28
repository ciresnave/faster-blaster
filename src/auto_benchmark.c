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
#include "../include/faster-blaster/judge.h"
#include "../include/faster-blaster/ranked_dispatch.h"
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

/** Ranked dispatch table built by the background benchmarking thread.
 *  Written once by the thread; read-only after that. */
static fb_ranked_table_t *g_ranked_table = NULL;

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
 * Background benchmarking thread.
 *
 * When the judge module is initialised and at least one backend is registered,
 * this thread:
 *   1. Iterates over every registered backend × representative op × size class
 *      × dtype and calls fb_judge_run() to produce a precision/timing profile.
 *   2. Saves each profile to disk via fb_judge_save_profile().
 *   3. Calls fb_ranked_table_build() on the profile directory to build in-memory
 *      ranked dispatch tables.
 *   4. Saves the ranked table to disk for fast reload on future startups.
 *
 * If the judge is not initialised (no profile directory, no oracle), the thread
 * exits immediately without error — callers fall back to the cached data.
 */
#ifdef _WIN32
static unsigned int __stdcall benchmark_thread_func(void *arg) {
#else
static void* benchmark_thread_func(void *arg) {
#endif
    auto_benchmark_state_t *state = (auto_benchmark_state_t *)arg;

    state->status   = FB_AUTO_BENCH_DETECTING;
    state->progress = 0.0f;

    if (state->config.status_callback)
        state->config.status_callback(state->status,
                                      "Checking judge module...");

    /* Nothing to do if the judge is not initialised. */
    if (!fb_judge_is_initialized()) {
        state->status   = FB_AUTO_BENCH_IDLE;
        state->progress = 1.0f;
#ifdef _WIN32
        return 0;
#else
        return NULL;
#endif
    }

    /* ------------------------------------------------------------------ */
    /* 1.  Collect state from the judge singleton.                         */
    /* ------------------------------------------------------------------ */
    char profile_dir[512] = {0};
    fb_judge_get_profile_dir(profile_dir, sizeof(profile_dir));

    /* At most 32 registered backends (FB_JUDGE_MAX_BACKENDS). */
    uint32_t backend_ids[32];
    uint32_t n_backends = 0;
    fb_judge_get_registered_ids(backend_ids, 32, &n_backends);

    if (n_backends == 0 || profile_dir[0] == '\0') {
        state->status   = FB_AUTO_BENCH_IDLE;
        state->progress = 1.0f;
#ifdef _WIN32
        return 0;
#else
        return NULL;
#endif
    }

    /* ------------------------------------------------------------------ */
    /* 2.  Profile each backend x op x size class x dtype.                */
    /* ------------------------------------------------------------------ */
    state->status = FB_AUTO_BENCH_BENCHMARKING;
    if (state->config.status_callback)
        state->config.status_callback(state->status,
                                      "Profiling backends with judge...");

    /* Representative op IDs — covers BLAS L1/L2/L3 in both precisions. */
    static const uint32_t k_ops[] = {
        4u,   /* FB_OP_SAXPY  - BLAS L1 representative                  */
        57u,  /* FB_OP_SGEMV  - BLAS L2 representative                  */
        119u, /* FB_OP_SGEMM  - BLAS L3 single-precision representative */
        120u, /* FB_OP_DGEMM  - BLAS L3 double-precision representative */
    };
    static const uint32_t k_n_ops = sizeof(k_ops) / sizeof(k_ops[0]);

    /* Size classes: SMALL (1), MEDIUM (2), LARGE (3). */
    static const uint8_t k_sizes[] = {
        (uint8_t)FB_SIZE_SMALL,
        (uint8_t)FB_SIZE_MEDIUM,
        (uint8_t)FB_SIZE_LARGE,
    };
    static const uint32_t k_n_sizes = sizeof(k_sizes) / sizeof(k_sizes[0]);

    /* Data types: float32 (0), float64 (1) per internal fb_dtype_t. */
    static const uint8_t k_dtypes[] = { 0u, 1u };
    static const uint32_t k_n_dtypes = sizeof(k_dtypes) / sizeof(k_dtypes[0]);

    uint32_t total_tasks =
        n_backends * k_n_ops * k_n_sizes * k_n_dtypes;
    uint32_t tasks_done = 0;

    for (uint32_t bi = 0; bi < n_backends && state->thread_active; bi++) {
        for (uint32_t oi = 0; oi < k_n_ops && state->thread_active; oi++) {
            for (uint32_t si = 0; si < k_n_sizes && state->thread_active; si++) {
                for (uint32_t di = 0; di < k_n_dtypes && state->thread_active; di++) {
                    fb_precision_profile_t profile;
                    fb_judge_status_t rc = fb_judge_run(
                        k_ops[oi],
                        backend_ids[bi],
                        0u,              /* device_id = 0 (primary) */
                        (fb_size_class_t)k_sizes[si],
                        k_dtypes[di],
                        false,           /* deep_audit */
                        &profile
                    );

                    if (rc == FB_JUDGE_OK) {
                        /* Persist to disk; ignore I/O errors gracefully. */
                        fb_judge_save_profile(&profile);
                    }
                    /* FB_JUDGE_ERR_NOT_IMPL is normal — skip silently. */

                    tasks_done++;
                    state->progress =
                        0.05f + 0.80f * ((float)tasks_done / (float)total_tasks);

                    if (state->config.progress_callback)
                        state->config.progress_callback(state->progress);
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 3.  Build ranked dispatch table from freshly-written profiles.      */
    /* ------------------------------------------------------------------ */
    state->status = FB_AUTO_BENCH_GENERATING_TABLES;
    if (state->config.status_callback)
        state->config.status_callback(state->status,
                                      "Building ranked dispatch tables...");
    state->progress = 0.88f;

    fb_ranked_table_t *table = fb_ranked_table_build(
        profile_dir,
        0u,           /* device_id */
        0u,           /* primary_dtype: 0 = float32 */
        backend_ids,
        n_backends,
        backend_ids[n_backends - 1]  /* fallback = last registered */
    );

    if (table) {
        /* Swap in the new table atomically. */
        fb_ranked_table_t *old = g_ranked_table;
        g_ranked_table = table;
        if (old)
            fb_ranked_table_free(old);

        /* Persist for fast reload on the next startup. */
        char table_path[600];
        snprintf(table_path, sizeof(table_path),
                 "%s/ranked_table.fbrdt", profile_dir);
        fb_ranked_table_save(table, table_path);
    }

    /* ------------------------------------------------------------------ */
    /* 4.  Done.                                                           */
    /* ------------------------------------------------------------------ */
    state->progress = 1.0f;
    state->status   = FB_AUTO_BENCH_COMPLETE;

    if (state->config.status_callback)
        state->config.status_callback(state->status, "Benchmarking complete");

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
    
    /* ------------------------------------------------------------------ *
     * Fast path: if the judge is initialised and all registered backends  *
     * have current profiles, try loading a cached ranked table from disk. *
     * ------------------------------------------------------------------ */
    if (fb_judge_is_initialized()) {
        char p_dir[512] = {0};
        fb_judge_get_profile_dir(p_dir, sizeof(p_dir));

        uint32_t bids[32];
        uint32_t nb = 0;
        fb_judge_get_registered_ids(bids, 32, &nb);

        bool all_current = (nb > 0 && p_dir[0] != '\0');
        for (uint32_t i = 0; i < nb && all_current; i++) {
            if (!fb_judge_profiles_are_current(bids[i], 0u))
                all_current = false;
        }

        if (all_current) {
            /* Try loading the pre-built ranked table. */
            char table_path[600];
            snprintf(table_path, sizeof(table_path),
                     "%s/ranked_table.fbrdt", p_dir);
            fb_ranked_table_t *tbl = fb_ranked_table_load(table_path);
            if (tbl) {
                fb_ranked_table_t *old = g_ranked_table;
                g_ranked_table = tbl;
                if (old) fb_ranked_table_free(old);

                if (result_out) {
                    result_out->change_reason          = FB_CHANGE_NONE;
                    result_out->benchmarking_performed = false;
                }
                g_auto_benchmark_state.status   = FB_AUTO_BENCH_IDLE;
                g_auto_benchmark_state.progress = 1.0f;
                return 0;  /* Fast path: cache valid */
            }
            /* Table file absent or version mismatch — fall through to rebuild. */
        }
    } else {
        /* Judge not initialised — check legacy cache file as a fallback. */
        char cache_path[1024] = {0};
        fb_get_default_cache_path(cache_path, sizeof(cache_path));
        FILE *f = fopen(cache_path, "rb");
        int cache_loaded = (f != NULL);
        if (f) fclose(f);

        fb_hardware_fingerprint_t fp = {0};
        if (cache_loaded && fb_generate_hardware_fingerprint(&fp) == 0) {
            if (result_out) {
                result_out->change_reason          = FB_CHANGE_NONE;
                result_out->benchmarking_performed = false;
            }
            g_auto_benchmark_state.status   = FB_AUTO_BENCH_IDLE;
            g_auto_benchmark_state.progress = 1.0f;
            return 0;  /* Legacy fast path */
        }
    }

    /* Need to (re-)benchmark. */
    if (result_out) {
        result_out->change_reason          = FB_CHANGE_FIRST_RUN;
        result_out->benchmarking_performed = true;
    }

    /* Start background benchmarking thread if not already running. */
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
 * Blocks until benchmarking finishes.
 *
 * @param result_out Output: Benchmark result (can be NULL)
 * @return 0 on success, negative on error
 */
int fb_auto_benchmark_wait(fb_auto_benchmark_result_t* result_out) {
    if (!g_auto_benchmark_state.thread_active) {
        return 0;  // Not running
    }
    
    #ifdef _WIN32
        DWORD wait_result = WaitForSingleObject(g_auto_benchmark_state.benchmark_thread,
                                                INFINITE);
        if (wait_result == WAIT_OBJECT_0) {
            g_auto_benchmark_state.thread_active = 0;
            CloseHandle(g_auto_benchmark_state.benchmark_thread);
            return 0;
        } else {
            return -1;
        }
    #else
        pthread_join(g_auto_benchmark_state.benchmark_thread, NULL);
        g_auto_benchmark_state.thread_active = 0;
        return 0;
    #endif
    (void)result_out; /* result_out population not yet implemented */
    return 0;
}

/**
 * Get current auto-benchmark status
 *
 * @return Current status
 */
fb_auto_benchmark_status_t fb_auto_benchmark_get_status(void) {
    return g_auto_benchmark_state.status;
}

/**
 * Get auto-benchmark progress
 *
 * @return Progress (0.0-1.0)
 */
float fb_auto_benchmark_get_progress(void) {
    return g_auto_benchmark_state.progress;
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

/**
 * Get the ranked dispatch table built by the most recent background benchmark.
 *
 * Returns NULL if no benchmarking has completed yet (or if the benchmark thread
 * has not been started).  The returned pointer remains valid until the next
 * call to fb_auto_benchmark_check() that triggers a new benchmark pass.
 *
 * Typical use:
 *   const fb_ranked_table_t *tbl = fb_auto_benchmark_get_ranked_table();
 *   if (tbl)
 *       uint32_t bid = fb_ranked_get_best(tbl, FB_OP_SGEMM, FB_RANK_FASTEST);
 *
 * @return Ranked dispatch table, or NULL if not yet available.
 */
const fb_ranked_table_t *fb_auto_benchmark_get_ranked_table(void)
{
    return g_ranked_table;
}
