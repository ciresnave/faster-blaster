/**
 * @file timing.c
 * @brief High-resolution timing implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "timing.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
    static LARGE_INTEGER g_frequency;
    static bool g_timing_initialized = false;
#elif defined(__APPLE__)
    #include <mach/mach_time.h>
    static mach_timebase_info_data_t g_timebase;
    static bool g_timing_initialized = false;
#else
    #include <time.h>
    #include <unistd.h>
#endif

int fb_timing_init(void) {
    #ifdef _WIN32
        if (!g_timing_initialized) {
            QueryPerformanceFrequency(&g_frequency);
            g_timing_initialized = true;
        }
    #elif defined(__APPLE__)
        if (!g_timing_initialized) {
            mach_timebase_info(&g_timebase);
            g_timing_initialized = true;
        }
    #endif
    return 0;
}

uint64_t fb_timing_now_ns(void) {
    #ifdef _WIN32
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return (uint64_t)((counter.QuadPart * 1000000000ULL) / g_frequency.QuadPart);
    #elif defined(__APPLE__)
        uint64_t abs_time = mach_absolute_time();
        return (abs_time * g_timebase.numer) / g_timebase.denom;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    #endif
}

void fb_timer_start(fb_timer_t *timer) {
    timer->start_time = fb_timing_now_ns();
    timer->running = true;
}

void fb_timer_stop(fb_timer_t *timer) {
    timer->end_time = fb_timing_now_ns();
    timer->running = false;
}

uint64_t fb_timer_elapsed_ns(const fb_timer_t *timer) {
    if (timer->running) {
        return fb_timing_now_ns() - timer->start_time;
    }
    return timer->end_time - timer->start_time;
}

double fb_timer_elapsed_us(const fb_timer_t *timer) {
    return fb_timer_elapsed_ns(timer) / 1000.0;
}

double fb_timer_elapsed_ms(const fb_timer_t *timer) {
    return fb_timer_elapsed_ns(timer) / 1000000.0;
}

void fb_timer_reset(fb_timer_t *timer) {
    timer->start_time = 0;
    timer->end_time = 0;
    timer->running = false;
}

fb_timing_stats_t* fb_timing_stats_create(size_t capacity) {
    fb_timing_stats_t *stats = (fb_timing_stats_t*)malloc(sizeof(fb_timing_stats_t));
    if (!stats) return NULL;
    
    stats->measurements = (double*)malloc(capacity * sizeof(double));
    if (!stats->measurements) {
        free(stats);
        return NULL;
    }
    
    stats->count = 0;
    stats->capacity = capacity;
    stats->min = 0.0;
    stats->max = 0.0;
    stats->median = 0.0;
    stats->mean = 0.0;
    stats->stddev = 0.0;
    
    return stats;
}

int fb_timing_stats_add(fb_timing_stats_t *stats, double time_ns) {
    if (stats->count >= stats->capacity) {
        /* Resize array */
        size_t new_capacity = stats->capacity * 2;
        double *new_measurements = (double*)realloc(stats->measurements, 
                                                     new_capacity * sizeof(double));
        if (!new_measurements) return -1;
        
        stats->measurements = new_measurements;
        stats->capacity = new_capacity;
    }
    
    stats->measurements[stats->count++] = time_ns;
    return 0;
}

/* Comparison function for qsort */
static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int fb_timing_stats_compute(fb_timing_stats_t *stats) {
    if (stats->count == 0) return -1;
    
    /* Sort measurements for median calculation */
    qsort(stats->measurements, stats->count, sizeof(double), compare_double);
    
    /* Min and max */
    stats->min = stats->measurements[0];
    stats->max = stats->measurements[stats->count - 1];
    
    /* Median */
    if (stats->count % 2 == 0) {
        stats->median = (stats->measurements[stats->count / 2 - 1] + 
                        stats->measurements[stats->count / 2]) / 2.0;
    } else {
        stats->median = stats->measurements[stats->count / 2];
    }
    
    /* Mean */
    double sum = 0.0;
    for (size_t i = 0; i < stats->count; i++) {
        sum += stats->measurements[i];
    }
    stats->mean = sum / stats->count;
    
    /* Standard deviation */
    double variance_sum = 0.0;
    for (size_t i = 0; i < stats->count; i++) {
        double diff = stats->measurements[i] - stats->mean;
        variance_sum += diff * diff;
    }
    stats->stddev = sqrt(variance_sum / stats->count);
    
    return 0;
}

void fb_timing_stats_free(fb_timing_stats_t *stats) {
    if (stats) {
        free(stats->measurements);
        free(stats);
    }
}

void fb_timing_sleep_ns(uint64_t ns) {
    #ifdef _WIN32
        HANDLE timer = CreateWaitableTimer(NULL, TRUE, NULL);
        if (timer) {
            LARGE_INTEGER li;
            li.QuadPart = -(LONGLONG)(ns / 100); /* 100-nanosecond intervals */
            SetWaitableTimer(timer, &li, 0, NULL, NULL, FALSE);
            WaitForSingleObject(timer, INFINITE);
            CloseHandle(timer);
        }
    #else
        struct timespec ts;
        ts.tv_sec = ns / 1000000000ULL;
        ts.tv_nsec = ns % 1000000000ULL;
        nanosleep(&ts, NULL);
    #endif
}

double fb_timing_calc_gflops(uint64_t flops, double time_ns) {
    if (time_ns <= 0.0) return 0.0;
    return (flops / time_ns);  /* Returns GFLOPS */
}

double fb_timing_calc_bandwidth(uint64_t bytes, double time_ns) {
    if (time_ns <= 0.0) return 0.0;
    return (bytes / time_ns);  /* Returns GB/s */
}
