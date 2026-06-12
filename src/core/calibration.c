/**
 * @file calibration.c
 * @brief Calibration system implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "calibration.h"
#include "test_data.h"
#include "timing.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* MSVC doesn't have aligned_alloc until C11, use _aligned_malloc instead */
#ifdef _MSC_VER
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

/* Default calibration sizes */
static const size_t DEFAULT_SIZES[] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192
};
#define NUM_DEFAULT_SIZES (sizeof(DEFAULT_SIZES) / sizeof(DEFAULT_SIZES[0]))

typedef struct fb_calibration_config {
    size_t num_sizes;
    size_t *sizes;
    size_t num_warmup;
    size_t num_iterations;
    fb_operation_id_t *operations;  /* Changed from fb_blas_op_t */
    size_t num_operations;
    double tolerance;
    fb_calibration_callback_t callback;
    void *callback_data;
} fb_calibration_config_t;

fb_calibration_config_t *fb_calibration_config_create(void) {
    fb_calibration_config_t *config = calloc(1, sizeof(fb_calibration_config_t));
    if (!config) return NULL;
    
    /* Set defaults */
    config->num_warmup = 3;
    config->num_iterations = 10;
    config->tolerance = 1e-5;
    
    /* Default sizes */
    config->num_sizes = NUM_DEFAULT_SIZES;
    config->sizes = malloc(sizeof(size_t) * NUM_DEFAULT_SIZES);
    if (config->sizes) {
        memcpy(config->sizes, DEFAULT_SIZES, sizeof(DEFAULT_SIZES));
    }
    
    /* Default operation: SGEMM */
    config->num_operations = 1;
    config->operations = malloc(sizeof(fb_operation_id_t));
    if (config->operations) {
        config->operations[0] = FB_OP_SGEMM;
    }
    
    return config;
}

void fb_calibration_config_destroy(fb_calibration_config_t *config) {
    if (!config) return;
    free(config->sizes);
    free(config->operations);
    free(config);
}

int fb_calibration_config_set_sizes(
    fb_calibration_config_t *config,
    const size_t *sizes,
    size_t num_sizes)
{
    if (!config || !sizes || num_sizes == 0) return -1;
    
    size_t *new_sizes = realloc(config->sizes, sizeof(size_t) * num_sizes);
    if (!new_sizes) return -1;
    
    config->sizes = new_sizes;
    config->num_sizes = num_sizes;
    memcpy(config->sizes, sizes, sizeof(size_t) * num_sizes);
    
    return 0;
}

int fb_calibration_config_set_dtypes(
    fb_calibration_config_t *config,
    const fb_dtype_t *dtypes,
    size_t num_dtypes)
{
    if (!config || !dtypes || num_dtypes == 0) return -1;
    
    fb_dtype_t *new_dtypes = realloc(config->dtypes, sizeof(fb_dtype_t) * num_dtypes);
    if (!new_dtypes) return -1;
    
    config->dtypes = new_dtypes;
    config->num_dtypes = num_dtypes;
    memcpy(config->dtypes, dtypes, sizeof(fb_dtype_t) * num_dtypes);
    
    return 0;
}

int fb_calibration_config_set_operations(
    fb_calibration_config_t *config,
    const fb_blas_op_t *operations,
    size_t num_operations)
{
    if (!config || !operations || num_operations == 0) return -1;
    
    fb_blas_op_t *new_ops = realloc(config->operations, sizeof(fb_blas_op_t) * num_operations);
    if (!new_ops) return -1;
    
    config->operations = new_ops;
    config->num_operations = num_operations;
    memcpy(config->operations, operations, sizeof(fb_blas_op_t) * num_operations);
    
    return 0;
}

void fb_calibration_config_set_iterations(
    fb_calibration_config_t *config,
    size_t warmup,
    size_t iterations)
{
    if (!config) return;
    config->num_warmup = warmup;
    config->num_iterations = iterations;
}

void fb_calibration_config_set_tolerance(
    fb_calibration_config_t *config,
    double tolerance)
{
    if (!config) return;
    config->tolerance = tolerance;
}

void fb_calibration_config_set_callback(
    fb_calibration_config_t *config,
    fb_calibration_callback_t callback,
    void *user_data)
{
    if (!config) return;
    config->callback = callback;
    config->callback_data = user_data;
}

static int benchmark_gemm(
    const fb_backend_vtable_t *backend,
    size_t size,
    fb_dtype_t dtype,
    size_t num_warmup,
    size_t num_iterations,
    fb_timing_stats_t *stats)
{
    size_t elem_size = fb_dtype_size(dtype);
    if (elem_size == 0) return -1;
    
    /* Allocate matrices */
    void *A = aligned_alloc(64, size * size * elem_size);
    void *B = aligned_alloc(64, size * size * elem_size);
    void *C = aligned_alloc(64, size * size * elem_size);
    
    if (!A || !B || !C) {
        free(A); free(B); free(C);
        return -1;
    }
    
    /* Generate test data */
    fb_test_generate_matrix(A, size, size, dtype, FB_TEST_RANDOM, 42);
    fb_test_generate_matrix(B, size, size, dtype, FB_TEST_RANDOM, 43);
    fb_test_generate_matrix(C, size, size, dtype, FB_TEST_ZEROS, 0);
    
    /* Warmup */
    for (size_t i = 0; i < num_warmup; i++) {
        if (dtype == FB_DTYPE_FLOAT32 && backend->sgemm) {
            backend->sgemm(
                FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                size, size, size, 1.0f, A, size, B, size, 0.0f, C, size
            );
        } else if (dtype == FB_DTYPE_FLOAT64 && backend->dgemm) {
            backend->dgemm(
                FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                size, size, size, 1.0, A, size, B, size, 0.0, C, size
            );
        }
    }
    
    /* Benchmark */
    uint *times = malloc(sizeof(uint) * num_iterations);
    if (!times) {
        free(A); free(B); free(C);
        return -1;
    }
    
    for (size_t i = 0; i < num_iterations; i++) {
        fb_timer_t timer;
        fb_timer_start(&timer);
        
        if (dtype == FB_DTYPE_FLOAT32 && backend->sgemm) {
            backend->sgemm(
                FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                size, size, size, 1.0f, A, size, B, size, 0.0f, C, size
            );
        } else if (dtype == FB_DTYPE_FLOAT64 && backend->dgemm) {
            backend->dgemm(
                FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
                size, size, size, 1.0, A, size, B, size, 0.0, C, size
            );
        }
        
        times[i] = fb_timer_stop(&timer);
    }
    
    fb_timing_compute_stats(times, num_iterations, stats);
    
    free(times);
    free(A); free(B); free(C);
    
    return 0;
}

fb_calibration_result_t *fb_calibration_run(
    const fb_backend_vtable_t *backend,
    const char *backend_name,
    const fb_calibration_config_t *config)
{
    if (!backend || !backend_name || !config) return NULL;
    
    size_t total_benchmarks = config->num_sizes * config->num_dtypes * config->num_operations;
    
    fb_calibration_result_t *result = calloc(1, sizeof(fb_calibration_result_t));
    if (!result) return NULL;
    
    result->backend_name = strdup(backend_name);
    result->num_benchmarks = total_benchmarks;
    result->benchmarks = calloc(total_benchmarks, sizeof(fb_benchmark_result_t));
    
    if (!result->benchmarks) {
        fb_calibration_result_destroy(result);
        return NULL;
    }
    
    size_t bench_idx = 0;
    
    for (size_t op_idx = 0; op_idx < config->num_operations; op_idx++) {
        for (size_t size_idx = 0; size_idx < config->num_sizes; size_idx++) {
            for (size_t dtype_idx = 0; dtype_idx < config->num_dtypes; dtype_idx++) {
                fb_blas_op_t op = config->operations[op_idx];
                size_t size = config->sizes[size_idx];
                fb_dtype_t dtype = config->dtypes[dtype_idx];
                
                fb_benchmark_result_t *bench = &result->benchmarks[bench_idx];
                bench->operation = op;
                bench->size = size;
                bench->dtype = dtype;
                
                /* Run benchmark based on operation type */
                int status = 0;
                if (op == FB_BLAS_GEMM) {
                    status = benchmark_gemm(
                        backend, size, dtype,
                        config->num_warmup, config->num_iterations,
                        &bench->timing
                    );
                }
                
                bench->passed = (status == 0);
                
                /* Calculate GFLOPS for GEMM: 2*N^3 flops */
                if (bench->passed && op == FB_BLAS_GEMM) {
                    uint flops = 2ULL * size * size * size;
                    bench->gflops = fb_timing_to_gflops(flops, bench->timing.median_ns);
                }
                
                /* Progress callback */
                if (config->callback) {
                    config->callback(
                        bench_idx + 1, total_benchmarks,
                        bench, config->callback_data
                    );
                }
                
                bench_idx++;
            }
        }
    }
    
    return result;
}

void fb_calibration_result_destroy(fb_calibration_result_t *result) {
    if (!result) return;
    free((void*)result->backend_name);
    free(result->benchmarks);
    free(result);
}

int fb_calibration_result_to_json(
    const fb_calibration_result_t *result,
    const char *filepath)
{
    if (!result || !filepath) return -1;
    
    FILE *f = fopen(filepath, "w");
    if (!f) return -1;
    
    fprintf(f, "{\n");
    fprintf(f, "  \"backend\": \"%s\",\n", result->backend_name);
    fprintf(f, "  \"num_benchmarks\": %zu,\n", result->num_benchmarks);
    fprintf(f, "  \"benchmarks\": [\n");
    
    for (size_t i = 0; i < result->num_benchmarks; i++) {
        const fb_benchmark_result_t *bench = &result->benchmarks[i];
        
        fprintf(f, "    {\n");
        fprintf(f, "      \"operation\": %d,\n", bench->operation);
        fprintf(f, "      \"size\": %zu,\n", bench->size);
        fprintf(f, "      \"dtype\": %d,\n", bench->dtype);
        fprintf(f, "      \"passed\": %s,\n", bench->passed ? "true" : "false");
        fprintf(f, "      \"gflops\": %.2f,\n", bench->gflops);
        fprintf(f, "      \"timing\": {\n");
        fprintf(f, "        \"min_ns\": %llu,\n", (unsigned long long)bench->timing.min_ns);
        fprintf(f, "        \"max_ns\": %llu,\n", (unsigned long long)bench->timing.max_ns);
        fprintf(f, "        \"median_ns\": %llu,\n", (unsigned long long)bench->timing.median_ns);
        fprintf(f, "        \"mean_ns\": %.0f,\n", bench->timing.mean_ns);
        fprintf(f, "        \"stddev_ns\": %.0f\n", bench->timing.stddev_ns);
        fprintf(f, "      }\n");
        fprintf(f, "    }%s\n", (i < result->num_benchmarks - 1) ? "," : "");
    }
    
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    
    fclose(f);
    return 0;
}

fb_calibration_result_t *fb_calibration_result_from_json(const char *filepath) {
    /* JSON parsing would require a library like cJSON or similar */
    /* For now, return NULL as placeholder */
    (void)filepath;
    return NULL;
}
