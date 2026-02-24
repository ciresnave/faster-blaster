#include "benchmark_system.h"
#include <math.h>

// Size classification thresholds
#define SIZE_TINY_MAX       32
#define SIZE_SMALL_MAX      256
#define SIZE_MEDIUM_MAX     2048
#define SIZE_LARGE_MAX      8192

// Shape classification ratios
#define SHAPE_SQUARE_RATIO       1.1    // Within 10% is square
#define SHAPE_EXTREME_RATIO      4.0    // >4x difference is extreme
#define SHAPE_MIDDLE_RATIO       2.0    // 2x-4x is "middle"

fb_size_class_t fb_classify_size(size_t m, size_t n, size_t k) {
    // Determine characteristic size (max dimension or geometric mean)
    size_t max_dim = m;
    if (n > max_dim) max_dim = n;
    if (k > max_dim) max_dim = k;
    
    // For matrices, use geometric mean of dimensions
    size_t char_size = max_dim;
    if (m > 0 && n > 0) {
        char_size = (size_t)sqrt((double)m * (double)n);
        if (k > 0) {
            // 3D problem (like GEMM with m×k * k×n)
            char_size = (size_t)cbrt((double)m * (double)n * (double)k);
        }
    }
    
    // Classify based on characteristic size
    if (char_size <= SIZE_TINY_MAX) {
        return FB_SIZE_TINY;
    } else if (char_size <= SIZE_SMALL_MAX) {
        return FB_SIZE_SMALL;
    } else if (char_size <= SIZE_MEDIUM_MAX) {
        return FB_SIZE_MEDIUM;
    } else if (char_size <= SIZE_LARGE_MAX) {
        return FB_SIZE_LARGE;
    } else {
        return FB_SIZE_HUGE;
    }
}

fb_shape_class_t fb_classify_shape(size_t m, size_t n) {
    if (m == 0 || n == 0) {
        return FB_SHAPE_SQUARE; // Default for invalid input
    }
    
    double ratio = (double)m / (double)n;
    
    // Check if square (within 10%)
    if (ratio >= (1.0 / SHAPE_SQUARE_RATIO) && ratio <= SHAPE_SQUARE_RATIO) {
        return FB_SHAPE_SQUARE;
    }
    
    // Tall matrix (m > n)
    if (ratio > 1.0) {
        if (ratio >= SHAPE_EXTREME_RATIO) {
            return FB_SHAPE_TALL_SKINNY;  // m >> n (m > 4n)
        } else if (ratio >= SHAPE_MIDDLE_RATIO) {
            return FB_SHAPE_SKINNY_MIDDLE; // 2n < m ≤ 4n
        } else {
            return FB_SHAPE_SQUARE; // Close enough to square
        }
    }
    
    // Wide matrix (n > m)
    else {
        ratio = 1.0 / ratio; // Invert for easier comparison
        if (ratio >= SHAPE_EXTREME_RATIO) {
            return FB_SHAPE_SHORT_WIDE;   // n >> m (n > 4m)
        } else if (ratio >= SHAPE_MIDDLE_RATIO) {
            return FB_SHAPE_FAT_MIDDLE;   // 2m < n ≤ 4m
        } else {
            return FB_SHAPE_SQUARE; // Close enough to square
        }
    }
}

void fb_get_representative_size(fb_size_class_t size_class, 
                                size_t* m_out, 
                                size_t* n_out, 
                                size_t* k_out) {
    // Representative sizes for each class
    // These are geometric midpoints of each range
    switch (size_class) {
        case FB_SIZE_TINY:
            *m_out = 32;
            *n_out = 32;
            *k_out = 32;
            break;
        
        case FB_SIZE_SMALL:
            *m_out = 128;    // Midpoint of 64-256
            *n_out = 128;
            *k_out = 128;
            break;
        
        case FB_SIZE_MEDIUM:
            *m_out = 1024;   // Midpoint of 512-2048
            *n_out = 1024;
            *k_out = 1024;
            break;
        
        case FB_SIZE_LARGE:
            *m_out = 5120;   // Midpoint of 4096-8192
            *n_out = 5120;
            *k_out = 5120;
            break;
        
        case FB_SIZE_HUGE:
            *m_out = 16384;  // Representative "huge" size
            *n_out = 16384;
            *k_out = 16384;
            break;
        
        default:
            *m_out = 1024;
            *n_out = 1024;
            *k_out = 1024;
            break;
    }
}

void fb_get_representative_shape(fb_shape_class_t shape_class,
                                 size_t base_size,
                                 size_t* m_out,
                                 size_t* n_out) {
    switch (shape_class) {
        case FB_SHAPE_SQUARE:
            *m_out = base_size;
            *n_out = base_size;
            break;
        
        case FB_SHAPE_TALL_SKINNY:
            // m = 5n (extreme tall)
            *m_out = base_size * 5;
            *n_out = base_size;
            break;
        
        case FB_SHAPE_SHORT_WIDE:
            // n = 5m (extreme wide)
            *m_out = base_size;
            *n_out = base_size * 5;
            break;
        
        case FB_SHAPE_SKINNY_MIDDLE:
            // m = 3n (moderate tall)
            *m_out = base_size * 3;
            *n_out = base_size;
            break;
        
        case FB_SHAPE_FAT_MIDDLE:
            // n = 3m (moderate wide)
            *m_out = base_size;
            *n_out = base_size * 3;
            break;
        
        default:
            *m_out = base_size;
            *n_out = base_size;
            break;
    }
}
