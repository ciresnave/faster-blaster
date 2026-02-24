/**
 * @file backend_loader.c
 * @brief Implementation of dynamic backend loading
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "backend_loader.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "faster-blaster/gpu_backend_trait.h"

#ifdef _WIN32
#include <windows.h>
#define FB_LOAD_LIBRARY(path) LoadLibraryA(path)
#define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#define FB_FREE_LIBRARY(handle) FreeLibrary((HMODULE)(handle))
typedef HMODULE fb_lib_handle_t;
#else
#include <dlfcn.h>
#define FB_LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
#define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define FB_FREE_LIBRARY(handle) dlclose(handle)
typedef void* fb_lib_handle_t;
#endif

/* GPU headers */
#ifdef FB_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

/* Note: HIP headers cannot be included alongside CUDA headers due to type conflicts.
 * For HIP GPU enumeration, use fb_detect_gpus() from gpu_detect.c instead.
 * The legacy HIP enumeration code here is disabled when CUDA is also enabled. */

/* CPU feature detection */
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

/* Include backend adapters */
#include "openblas_backend.h"
#include "mkl_backend.h"

/* Forward declarations for backend-specific loaders */
static int load_openblas_backend(fb_backend_vtable_t* vtable);
static int load_mkl_backend(fb_backend_vtable_t* vtable);
static int load_accelerate_backend(fb_backend_vtable_t* vtable);
static int load_aocl_backend(fb_backend_vtable_t* vtable);
static int load_blis_backend(fb_backend_vtable_t* vtable);
static int load_cublas_backend(fb_backend_vtable_t* vtable);
static int load_rocblas_backend(fb_backend_vtable_t* vtable);
static int load_onemkl_gpu_backend(fb_backend_vtable_t* vtable);

/* Reference backend (always available) */
extern const fb_backend_vtable_t fb_reference_vtable;

/* Global backend registry */
static fb_backend_metadata_t g_backends[FB_BACKEND_COUNT];
static bool g_loader_initialized = false;
static fb_backend_type_t g_current_backend = FB_BACKEND_REFERENCE;
static char g_last_error[512] = {0};

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

static void detect_backend_availability(void) {
    /* Reference backend - always available */
    g_backends[FB_BACKEND_REFERENCE] = (fb_backend_metadata_t){
        .type = FB_BACKEND_REFERENCE,
        .name = "Reference",
        .version = "1.0.0",
        .license = "MIT",
        .vendor = "Faster-Blaster",
        .capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_LAPACK | FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX,
        .available = true,
        .loaded = false,
        .handle = NULL,
        .priority = 1
    };
    
    /* OpenBLAS - check for library */
    const char* openblas_libs[] = {
#ifdef _WIN32
        "libopenblas.dll", "openblas.dll"
#elif __APPLE__
        "libopenblas.dylib", "libopenblas.0.dylib"
#else
        "libopenblas.so", "libopenblas.so.0"
#endif
    };
    
    bool openblas_found = false;
    for (size_t i = 0; i < sizeof(openblas_libs) / sizeof(openblas_libs[0]); i++) {
        fb_lib_handle_t handle = FB_LOAD_LIBRARY(openblas_libs[i]);
        if (handle) {
            openblas_found = true;
            FB_FREE_LIBRARY(handle);
            break;
        }
    }
    
    g_backends[FB_BACKEND_OPENBLAS] = (fb_backend_metadata_t){
        .type = FB_BACKEND_OPENBLAS,
        .name = "OpenBLAS",
        .version = "0.3.x",
        .license = "BSD-3-Clause",
        .vendor = "OpenBLAS Project",
        .capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_LAPACK | FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX | 
                       FB_CAP_THREADSAFE,
        .available = openblas_found,
        .loaded = false,
        .handle = NULL,
        .priority = 50
    };
    
    /* Intel MKL - check for library */
    const char* mkl_libs[] = {
#ifdef _WIN32
        "mkl_rt.2.dll", "mkl_rt.dll"
#elif __APPLE__
        "libmkl_rt.dylib", "libmkl_rt.2.dylib"
#else
        "libmkl_rt.so", "libmkl_rt.so.2"
#endif
    };
    
    bool mkl_found = false;
    for (size_t i = 0; i < sizeof(mkl_libs) / sizeof(mkl_libs[0]); i++) {
        fb_lib_handle_t handle = FB_LOAD_LIBRARY(mkl_libs[i]);
        if (handle) {
            mkl_found = true;
            FB_FREE_LIBRARY(handle);
            break;
        }
    }
    
    g_backends[FB_BACKEND_MKL] = (fb_backend_metadata_t){
        .type = FB_BACKEND_MKL,
        .name = "Intel MKL",
        .version = "2024.x",
        .license = "Intel Simplified Software License",
        .vendor = "Intel Corporation",
        .capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_LAPACK | FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX | 
                       FB_CAP_THREADSAFE,
        .available = mkl_found,
        .loaded = false,
        .handle = NULL,
        .priority = 100  /* Highest priority for Intel CPUs */
    };
    
#ifdef __APPLE__
    /* Apple Accelerate - system framework on macOS/iOS */
    g_backends[FB_BACKEND_ACCELERATE] = (fb_backend_metadata_t){
        .type = FB_BACKEND_ACCELERATE,
        .name = "Apple Accelerate",
        .version = "System",
        .license = "Apple System Framework",
        .vendor = "Apple Inc.",
        .capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_LAPACK | FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX | 
                       FB_CAP_THREADSAFE,
        .available = true,
        .loaded = false,
        .handle = NULL,
        .priority = 90
    };
#else
    g_backends[FB_BACKEND_ACCELERATE] = (fb_backend_metadata_t){
        .type = FB_BACKEND_ACCELERATE,
        .name = "Apple Accelerate",
        .available = false,
        .priority = 0
    };
#endif
    
    /* AMD AOCL - check for library */
    const char* aocl_libs[] = {
#ifdef _WIN32
        "libaocl.dll", "aocl.dll"
#elif __APPLE__
        "libaocl.dylib"
#else
        "libaocl.so"
#endif
    };
    
    bool aocl_found = false;
    for (size_t i = 0; i < sizeof(aocl_libs) / sizeof(aocl_libs[0]); i++) {
        fb_lib_handle_t handle = FB_LOAD_LIBRARY(aocl_libs[i]);
        if (handle) {
            aocl_found = true;
            FB_FREE_LIBRARY(handle);
            break;
        }
    }
    
    g_backends[FB_BACKEND_AOCL] = (fb_backend_metadata_t){
        .type = FB_BACKEND_AOCL,
        .name = "AMD AOCL",
        .version = "4.x",
        .license = "BSD-3-Clause",
        .vendor = "AMD",
        .capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_LAPACK | FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX,
        .available = aocl_found,
        .loaded = false,
        .handle = NULL,
        .priority = 85
    };
    
    /* BLIS - check for library */
    const char* blis_libs[] = {
#ifdef _WIN32
        "libblis.dll", "blis.dll"
#elif __APPLE__
        "libblis.dylib", "libblis.4.dylib"
#else
        "libblis.so", "libblis.so.4"
#endif
    };
    
    bool blis_found = false;
    for (size_t i = 0; i < sizeof(blis_libs) / sizeof(blis_libs[0]); i++) {
        fb_lib_handle_t handle = FB_LOAD_LIBRARY(blis_libs[i]);
        if (handle) {
            blis_found = true;
            FB_FREE_LIBRARY(handle);
            break;
        }
    }
    
    g_backends[FB_BACKEND_BLIS] = (fb_backend_metadata_t){
        .type = FB_BACKEND_BLIS,
        .name = "BLIS",
        .version = "1.x",
        .license = "BSD-3-Clause",
        .vendor = "FLAME Project",
        .capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX,
        .available = blis_found,
        .loaded = false,
        .handle = NULL,
        .priority = 70
    };
    
    /* NVIDIA cuBLAS - check for CUDA runtime */
    const char* cublas_libs[] = {
#ifdef _WIN32
        "cublas64_12.dll", "cublas64_11.dll"
#else
        "libcublas.so.12", "libcublas.so.11"
#endif
    };
    
    bool cublas_found = false;
    for (size_t i = 0; i < sizeof(cublas_libs) / sizeof(cublas_libs[0]); i++) {
        fb_lib_handle_t handle = FB_LOAD_LIBRARY(cublas_libs[i]);
        if (handle) {
            cublas_found = true;
            FB_FREE_LIBRARY(handle);
            break;
        }
    }
    
    g_backends[FB_BACKEND_CUBLAS] = (fb_backend_metadata_t){
        .type = FB_BACKEND_CUBLAS,
        .name = "NVIDIA cuBLAS",
        .version = "12.x",
        .license = "NVIDIA CUDA EULA",
        .vendor = "NVIDIA Corporation",
        .capabilities = FB_CAP_GPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX,
        .available = cublas_found,
        .loaded = false,
        .handle = NULL,
        .priority = 200  /* GPUs get highest priority */
    };
    
    /* AMD rocBLAS - check for ROCm */
    const char* rocblas_libs[] = {
#ifdef _WIN32
        "rocblas.dll"
#else
        "librocblas.so", "librocblas.so.4"
#endif
    };
    
    bool rocblas_found = false;
    for (size_t i = 0; i < sizeof(rocblas_libs) / sizeof(rocblas_libs[0]); i++) {
        fb_lib_handle_t handle = FB_LOAD_LIBRARY(rocblas_libs[i]);
        if (handle) {
            rocblas_found = true;
            FB_FREE_LIBRARY(handle);
            break;
        }
    }
    
    g_backends[FB_BACKEND_ROCBLAS] = (fb_backend_metadata_t){
        .type = FB_BACKEND_ROCBLAS,
        .name = "AMD rocBLAS",
        .version = "4.x",
        .license = "MIT",
        .vendor = "AMD",
        .capabilities = FB_CAP_GPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | 
                       FB_CAP_DOUBLE | FB_CAP_SINGLE | FB_CAP_COMPLEX,
        .available = rocblas_found,
        .loaded = false,
        .handle = NULL,
        .priority = 190
    };
    
    /* Intel oneMKL GPU */
    g_backends[FB_BACKEND_ONEMKL_GPU] = (fb_backend_metadata_t){
        .type = FB_BACKEND_ONEMKL_GPU,
        .name = "Intel oneMKL GPU",
        .version = "2024.x",
        .license = "Apache 2.0 + Intel Runtime",
        .vendor = "Intel Corporation",
        .capabilities = FB_CAP_GPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3,
        .available = false,  /* TODO: Implement detection */
        .loaded = false,
        .handle = NULL,
        .priority = 180
    };
    
    /* Custom backend slot */
    g_backends[FB_BACKEND_CUSTOM] = (fb_backend_metadata_t){
        .type = FB_BACKEND_CUSTOM,
        .name = "Custom",
        .available = false,
        .priority = 0
    };
}

int fb_backend_loader_init(void) {
    if (g_loader_initialized) {
        return 0;
    }
    
    memset(g_backends, 0, sizeof(g_backends));
    detect_backend_availability();
    
    g_current_backend = FB_BACKEND_REFERENCE;
    g_loader_initialized = true;
    
    return 0;
}

void fb_backend_loader_shutdown(void) {
    if (!g_loader_initialized) {
        return;
    }
    
    /* Unload all loaded backends */
    for (int i = 0; i < FB_BACKEND_COUNT; i++) {
        if (g_backends[i].loaded && g_backends[i].handle) {
            FB_FREE_LIBRARY((fb_lib_handle_t)g_backends[i].handle);
            g_backends[i].loaded = false;
            g_backends[i].handle = NULL;
        }
    }
    
    g_loader_initialized = false;
}

int fb_backend_get_info(fb_backend_type_t type, fb_backend_metadata_t* info) {
    if (!g_loader_initialized) {
        set_error("Backend loader not initialized");
        return -1;
    }
    
    if (type < 0 || type >= FB_BACKEND_COUNT) {
        set_error("Invalid backend type: %d", type);
        return -1;
    }
    
    if (!info) {
        set_error("NULL info parameter");
        return -1;
    }
    
    *info = g_backends[type];
    return 0;
}

int fb_backend_list_available(fb_backend_metadata_t* infos, int max_count) {
    if (!g_loader_initialized) {
        return 0;
    }
    
    int count = 0;
    for (int i = 0; i < FB_BACKEND_COUNT; i++) {
        if (g_backends[i].available) {
            if (infos && count < max_count) {
                infos[count] = g_backends[i];
            }
            count++;
        }
    }
    
    return count;
}

int fb_backend_load(fb_backend_type_t type, fb_backend_vtable_t* vtable) {
    if (!g_loader_initialized) {
        set_error("Backend loader not initialized");
        return -1;
    }
    
    if (type < 0 || type >= FB_BACKEND_COUNT) {
        set_error("Invalid backend type: %d", type);
        return -1;
    }
    
    if (!g_backends[type].available) {
        set_error("Backend '%s' not available", g_backends[type].name);
        return -1;
    }
    
    if (!vtable) {
        set_error("NULL vtable parameter");
        return -1;
    }
    
    /* If already loaded, just return the vtable */
    if (g_backends[type].loaded) {
        /* TODO: Return cached vtable */
        return 0;
    }
    
    int result = -1;
    switch (type) {
        case FB_BACKEND_REFERENCE:
            *vtable = fb_reference_vtable;
            result = 0;
            break;
        case FB_BACKEND_OPENBLAS:
            result = load_openblas_backend(vtable);
            break;
        case FB_BACKEND_MKL:
            result = load_mkl_backend(vtable);
            break;
        case FB_BACKEND_ACCELERATE:
            result = load_accelerate_backend(vtable);
            break;
        case FB_BACKEND_AOCL:
            result = load_aocl_backend(vtable);
            break;
        case FB_BACKEND_BLIS:
            result = load_blis_backend(vtable);
            break;
        case FB_BACKEND_CUBLAS:
            result = load_cublas_backend(vtable);
            break;
        case FB_BACKEND_ROCBLAS:
            result = load_rocblas_backend(vtable);
            break;
        case FB_BACKEND_ONEMKL_GPU:
            result = load_onemkl_gpu_backend(vtable);
            break;
        default:
            set_error("Backend loading not implemented for type %d", type);
            return -1;
    }
    
    if (result == 0) {
        g_backends[type].loaded = true;
    }
    
    return result;
}

void fb_backend_unload(fb_backend_type_t type) {
    if (!g_loader_initialized || type < 0 || type >= FB_BACKEND_COUNT) {
        return;
    }
    
    if (g_backends[type].loaded && g_backends[type].handle) {
        FB_FREE_LIBRARY((fb_lib_handle_t)g_backends[type].handle);
        g_backends[type].loaded = false;
        g_backends[type].handle = NULL;
    }
}

fb_backend_type_t fb_backend_auto_select(bool prefer_gpu) {
    if (!g_loader_initialized) {
        return FB_BACKEND_REFERENCE;
    }
    
    fb_backend_type_t best = FB_BACKEND_REFERENCE;
    int best_priority = -1;
    
    for (int i = 0; i < FB_BACKEND_COUNT; i++) {
        if (!g_backends[i].available) {
            continue;
        }
        
        /* Filter by GPU preference */
        bool is_gpu = (g_backends[i].capabilities & FB_CAP_GPU) != 0;
        if (prefer_gpu && !is_gpu) {
            continue;
        }
        
        if (g_backends[i].priority > best_priority) {
            best_priority = g_backends[i].priority;
            best = g_backends[i].type;
        }
    }
    
    return best;
}

int fb_backend_detect_cpu(char* vendor, uint64_t* features) {
    if (!vendor || !features) {
        return -1;
    }
    
    *features = 0;
    
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
    int cpuinfo[4] = {0};
    
#if defined(_MSC_VER)
    __cpuid(cpuinfo, 0);
#else
    __cpuid(0, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
    
    /* Extract vendor string */
    memcpy(vendor, &cpuinfo[1], 4);
    memcpy(vendor + 4, &cpuinfo[3], 4);
    memcpy(vendor + 8, &cpuinfo[2], 4);
    vendor[12] = '\0';
    
    /* Get feature flags */
#if defined(_MSC_VER)
    __cpuid(cpuinfo, 1);
#else
    __cpuid(1, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
    
    /* Parse common CPU features (simplified) */
    if (cpuinfo[3] & (1 << 25)) *features |= (1ULL << 0);  /* SSE */
    if (cpuinfo[3] & (1 << 26)) *features |= (1ULL << 1);  /* SSE2 */
    if (cpuinfo[2] & (1 << 0))  *features |= (1ULL << 2);  /* SSE3 */
    if (cpuinfo[2] & (1 << 9))  *features |= (1ULL << 3);  /* SSSE3 */
    if (cpuinfo[2] & (1 << 19)) *features |= (1ULL << 4);  /* SSE4.1 */
    if (cpuinfo[2] & (1 << 20)) *features |= (1ULL << 5);  /* SSE4.2 */
    if (cpuinfo[2] & (1 << 28)) *features |= (1ULL << 6);  /* AVX */
    
#else
    strncpy(vendor, "Unknown", 64);
#endif
    
    return 0;
}

int fb_backend_query_gpu_devices(fb_gpu_device_info_t* devices, int max_devices) {
    int total_devices = 0;
    
#ifdef FB_ENABLE_CUDA
    /* Query CUDA devices */
    int cuda_device_count = 0;
    if (cudaGetDeviceCount(&cuda_device_count) == cudaSuccess && cuda_device_count > 0) {
        for (int i = 0; i < cuda_device_count && total_devices < max_devices; i++) {
            struct cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                if (devices) {
                    fb_gpu_device_info_t* dev = &devices[total_devices];
                    dev->device_id = total_devices;
                    strncpy(dev->name, prop.name, sizeof(dev->name) - 1);
                    dev->total_memory = prop.totalGlobalMem;
                    dev->free_memory = 0;  /* Query with cudaMemGetInfo if needed */
                    dev->compute_capability_major = prop.major;
                    dev->compute_capability_minor = prop.minor;
                    dev->backend_type = FB_BACKEND_CUBLAS;
                }
                total_devices++;
            }
        }
    }
#endif
    
#if defined(FB_ENABLE_HIP) && !defined(FB_ENABLE_CUDA)
    /* Query ROCm devices */
    /* Note: This legacy API only works when CUDA is NOT also enabled (header conflicts).
     * For multi-GPU platform detection, use fb_detect_gpus() from gpu_detect.c instead. */
    #error "HIP enumeration via backend_loader requires HIP headers which conflict with CUDA. Use fb_detect_gpus() instead."
#endif
    
#ifdef FB_ENABLE_ONEMKL
    /* Query Intel Level Zero devices */
    /* TODO: Implement Level Zero device enumeration */
#endif
    
    return total_devices;
}

int fb_backend_register_custom(const char* name, const fb_backend_vtable_t* vtable, 
                                uint32_t capabilities) {
    if (!g_loader_initialized) {
        set_error("Backend loader not initialized");
        return -1;
    }
    
    if (!name || !vtable) {
        set_error("NULL parameter");
        return -1;
    }
    
    g_backends[FB_BACKEND_CUSTOM].name = name;
    g_backends[FB_BACKEND_CUSTOM].available = true;
    g_backends[FB_BACKEND_CUSTOM].capabilities = capabilities;
    g_backends[FB_BACKEND_CUSTOM].priority = 60;  /* Medium priority */
    
    /* TODO: Store custom vtable */
    
    return 0;
}

fb_backend_type_t fb_backend_get_current(void) {
    return g_current_backend;
}

int fb_backend_set_current(fb_backend_type_t type) {
    if (!g_loader_initialized) {
        set_error("Backend loader not initialized");
        return -1;
    }
    
    if (type < 0 || type >= FB_BACKEND_COUNT) {
        set_error("Invalid backend type: %d", type);
        return -1;
    }
    
    if (!g_backends[type].available) {
        set_error("Backend '%s' not available", g_backends[type].name);
        return -1;
    }
    
    g_current_backend = type;
    return 0;
}

const char* fb_backend_get_error(void) {
    return g_last_error;
}

/* Backend-specific loaders */
static int load_openblas_backend(fb_backend_vtable_t* vtable) {
    if (fb_openblas_init() != 0) {
        set_error("Failed to initialize OpenBLAS backend");
        return -1;
    }
    
    const fb_backend_vtable_t* openblas_vtable = fb_openblas_get_vtable();
    if (!openblas_vtable) {
        set_error("Failed to get OpenBLAS vtable");
        return -1;
    }
    
    *vtable = *openblas_vtable;
    return 0;
}

static int load_mkl_backend(fb_backend_vtable_t* vtable) {
    if (fb_mkl_init() != 0) {
        set_error("Failed to initialize MKL backend");
        return -1;
    }
    
    const fb_backend_vtable_t* mkl_vtable = fb_mkl_get_vtable();
    if (!mkl_vtable) {
        set_error("Failed to get MKL vtable");
        return -1;
    }
    
    *vtable = *mkl_vtable;
    return 0;
}

static int load_accelerate_backend(fb_backend_vtable_t* vtable) {
    set_error("Accelerate backend loader not yet implemented");
    return -1;
}

static int load_aocl_backend(fb_backend_vtable_t* vtable) {
    set_error("AOCL backend loader not yet implemented");
    return -1;
}

static int load_blis_backend(fb_backend_vtable_t* vtable) {
    set_error("BLIS backend loader not yet implemented");
    return -1;
}

static int load_cublas_backend(fb_backend_vtable_t* vtable) {
#ifdef FB_ENABLE_CUDA
    extern const fb_gpu_backend_trait_t* fb_cublas_get_backend(void);
    const fb_gpu_backend_trait_t* cublas = fb_cublas_get_backend();
    if (!cublas) {
        set_error("Failed to get cuBLAS backend trait");
        return -1;
    }
    
    /* Wrap GPU trait into generic vtable */
    /* TODO: Implement GPU trait to vtable wrapper */
    set_error("cuBLAS vtable wrapper not yet implemented");
    return -1;
#else
    set_error("cuBLAS backend not compiled in (FB_ENABLE_CUDA=OFF)");
    return -1;
#endif
}

static int load_rocblas_backend(fb_backend_vtable_t* vtable) {
#if defined(FB_ENABLE_HIP) && defined(FB_HAVE_ROCBLAS_BACKEND)
    extern const fb_gpu_backend_trait_t* fb_rocblas_get_backend(void);
    const fb_gpu_backend_trait_t* rocblas = fb_rocblas_get_backend();
    if (!rocblas) {
        set_error("Failed to get rocBLAS backend trait");
        return -1;
    }
    
    /* Wrap GPU trait into generic vtable */
    /* TODO: Implement GPU trait to vtable wrapper */
    set_error("rocBLAS vtable wrapper not yet implemented");
    return -1;
#else
    (void)vtable;
    set_error("rocBLAS backend not compiled in (needs CUDA->HIP conversion fixes)");
    return -1;
#endif
}

static int load_onemkl_gpu_backend(fb_backend_vtable_t* vtable) {
#ifdef FB_ENABLE_ONEMKL
    extern const fb_gpu_backend_trait_t* fb_onemkl_gpu_get_backend(void);
    const fb_gpu_backend_trait_t* onemkl = fb_onemkl_gpu_get_backend();
    if (!onemkl) {
        set_error("Failed to get oneMKL GPU backend trait");
        return -1;
    }
    
    /* Wrap GPU trait into generic vtable */
    /* TODO: Implement GPU trait to vtable wrapper */
    set_error("oneMKL GPU vtable wrapper not yet implemented");
    return -1;
#else
    set_error("oneMKL GPU backend not compiled in (FB_ENABLE_ONEMKL=OFF)");
    return -1;
#endif
}
