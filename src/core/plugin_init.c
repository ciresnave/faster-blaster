/**
 * @file plugin_init.c
 * @brief Central plugin registration - registers all compiled-in plugins
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include <stdio.h>

/* External plugin registration function declarations */
extern void fb_register_aocl_plugin(void);
extern void fb_register_standard_blis_plugin(void);
extern void fb_register_openblas_plugin(void);
extern void fb_register_mkl_plugin(void);
extern void fb_register_accelerate_plugin(void);
extern void fb_register_cublas_plugin(void);
extern void fb_register_rocblas_plugin(void);
extern void fb_register_onemkl_plugin(void);
extern void fb_register_metal_plugin(void);
extern void fb_register_clblast_plugin(void);
extern void fb_register_clblas_plugin(void);

/**
 * Register all compiled-in plugins
 * Called during fb_init()
 */
void fb_register_all_plugins(void) {
    /* CPU Plugins */
    #ifdef FB_ENABLE_AOCL
    fb_register_aocl_plugin();
    #endif
    
    #ifdef FB_ENABLE_BLIS
    fb_register_standard_blis_plugin();
    #endif
    
    #ifdef FB_ENABLE_OPENBLAS
    fb_register_openblas_plugin();
    #endif
    
    #ifdef FB_ENABLE_MKL
    fb_register_mkl_plugin();
    #endif
    
    #ifdef FB_ENABLE_ACCELERATE
    fb_register_accelerate_plugin();
    #endif
    
    /* GPU Plugins */
    #ifdef FB_ENABLE_CUDA
    fb_register_cublas_plugin();
    #endif
    
    #ifdef FB_ENABLE_ROCM
    fb_register_rocblas_plugin();
    #endif
    
    #ifdef FB_ENABLE_ONEMKL
    fb_register_onemkl_plugin();
    #endif
    
    #ifdef FB_ENABLE_METAL
    fb_register_metal_plugin();
    #endif
        #ifdef FB_ENABLE_OPENCL
    fb_register_clblast_plugin();
    fb_register_clblas_plugin();
    #endif
        printf("[INFO] Plugin registration complete\n");
}
