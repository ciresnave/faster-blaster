/**
 * @file clblast_trait_impl.c
 * @brief CLBlast OpenCL implementation of GPU backend trait
 * 
 * Provides CLBlast-specific implementation of the unified GPU backend trait interface.
 * This allows transparent use of OpenCL GPUs (AMD, NVIDIA, Intel, etc.) alongside 
 * vendor-specific implementations.
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * CLBlast Function Pointer Types
 * ========================================================================== */

// CLBlast enums
typedef enum {
    CLBlastLayoutRowMajor = 101,
    CLBlastLayoutColMajor = 102
} CLBlastLayout;

typedef enum {
    CLBlastTransposeNo = 111,
    CLBlastTransposeYes = 112,
    CLBlastTransposeConjugate = 113
} CLBlastTranspose;

typedef enum {
    CLBlastSideLeft = 141,
    CLBlastSideRight = 142
} CLBlastSide;

typedef enum {
    CLBlastTriangleUpper = 121,
    CLBlastTriangleLower = 122
} CLBlastTriangle;

typedef enum {
    CLBlastDiagonalNonUnit = 131,
    CLBlastDiagonalUnit = 132
} CLBlastDiagonal;

// Level 1 function pointers
typedef cl_int (*clblastSaxpy_t)(const size_t n, const float alpha,
                                  const cl_mem x, const size_t offx, const size_t incx,
                                  cl_mem y, const size_t offy, const size_t incy,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDaxpy_t)(const size_t n, const double alpha,
                                  const cl_mem x, const size_t offx, const size_t incx,
                                  cl_mem y, const size_t offy, const size_t incy,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastSdot_t)(const size_t n,
                                 cl_mem dot_buffer, const size_t dot_offset,
                                 const cl_mem x, const size_t offx, const size_t incx,
                                 const cl_mem y, const size_t offy, const size_t incy,
                                 cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDdot_t)(const size_t n,
                                 cl_mem dot_buffer, const size_t dot_offset,
                                 const cl_mem x, const size_t offx, const size_t incx,
                                 const cl_mem y, const size_t offy, const size_t incy,
                                 cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastScopy_t)(const size_t n,
                                  const cl_mem x, const size_t offx, const size_t incx,
                                  cl_mem y, const size_t offy, const size_t incy,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDcopy_t)(const size_t n,
                                  const cl_mem x, const size_t offx, const size_t incx,
                                  cl_mem y, const size_t offy, const size_t incy,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastSscal_t)(const size_t n, const float alpha,
                                  cl_mem x, const size_t offx, const size_t incx,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDscal_t)(const size_t n, const double alpha,
                                  cl_mem x, const size_t offx, const size_t incx,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastSnrm2_t)(const size_t n,
                                  cl_mem nrm2_buffer, const size_t nrm2_offset,
                                  const cl_mem x, const size_t offx, const size_t incx,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDnrm2_t)(const size_t n,
                                  cl_mem nrm2_buffer, const size_t nrm2_offset,
                                  const cl_mem x, const size_t offx, const size_t incx,
                                  cl_command_queue* queue, cl_event* event);

// Level 2 function pointers
typedef cl_int (*clblastSgemv_t)(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                  const size_t m, const size_t n, const float alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  const cl_mem x, const size_t x_offset, const size_t x_inc,
                                  const float beta,
                                  cl_mem y, const size_t y_offset, const size_t y_inc,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDgemv_t)(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                  const size_t m, const size_t n, const double alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  const cl_mem x, const size_t x_offset, const size_t x_inc,
                                  const double beta,
                                  cl_mem y, const size_t y_offset, const size_t y_inc,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastSger_t)(const CLBlastLayout layout,
                                 const size_t m, const size_t n, const float alpha,
                                 const cl_mem x, const size_t x_offset, const size_t x_inc,
                                 const cl_mem y, const size_t y_offset, const size_t y_inc,
                                 cl_mem a, const size_t a_offset, const size_t a_ld,
                                 cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDger_t)(const CLBlastLayout layout,
                                 const size_t m, const size_t n, const double alpha,
                                 const cl_mem x, const size_t x_offset, const size_t x_inc,
                                 const cl_mem y, const size_t y_offset, const size_t y_inc,
                                 cl_mem a, const size_t a_offset, const size_t a_ld,
                                 cl_command_queue* queue, cl_event* event);

// Level 3 function pointers
typedef cl_int (*clblastSgemm_t)(const CLBlastLayout layout,
                                  const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                  const size_t m, const size_t n, const size_t k,
                                  const float alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  const cl_mem b, const size_t b_offset, const size_t b_ld,
                                  const float beta,
                                  cl_mem c, const size_t c_offset, const size_t c_ld,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDgemm_t)(const CLBlastLayout layout,
                                  const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                  const size_t m, const size_t n, const size_t k,
                                  const double alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  const cl_mem b, const size_t b_offset, const size_t b_ld,
                                  const double beta,
                                  cl_mem c, const size_t c_offset, const size_t c_ld,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastSsyrk_t)(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                  const CLBlastTranspose a_transpose,
                                  const size_t n, const size_t k, const float alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  const float beta,
                                  cl_mem c, const size_t c_offset, const size_t c_ld,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDsyrk_t)(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                  const CLBlastTranspose a_transpose,
                                  const size_t n, const size_t k, const double alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  const double beta,
                                  cl_mem c, const size_t c_offset, const size_t c_ld,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastStrsm_t)(const CLBlastLayout layout, const CLBlastSide side,
                                  const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                  const CLBlastDiagonal diagonal,
                                  const size_t m, const size_t n, const float alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  cl_mem b, const size_t b_offset, const size_t b_ld,
                                  cl_command_queue* queue, cl_event* event);

typedef cl_int (*clblastDtrsm_t)(const CLBlastLayout layout, const CLBlastSide side,
                                  const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                  const CLBlastDiagonal diagonal,
                                  const size_t m, const size_t n, const double alpha,
                                  const cl_mem a, const size_t a_offset, const size_t a_ld,
                                  cl_mem b, const size_t b_offset, const size_t b_ld,
                                  cl_command_queue* queue, cl_event* event);

/* ============================================================================
 * CLBlast Context
 * ========================================================================== */

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    int device_id;
    void* lib_handle;  // DLL handle for dynamic loading
    
    // Function pointers (loaded from clblast.dll)
    clblastSaxpy_t saxpy;
    clblastDaxpy_t daxpy;
    clblastScopy_t scopy;
    clblastDcopy_t dcopy;
    clblastSscal_t sscal;
    clblastDscal_t dscal;
    clblastSdot_t sdot;
    clblastDdot_t ddot;
    clblastSnrm2_t snrm2;
    clblastDnrm2_t dnrm2;
    clblastSgemv_t sgemv;
    clblastDgemv_t dgemv;
    clblastSger_t sger;
    clblastDger_t dger;
    clblastSgemm_t sgemm;
    clblastDgemm_t dgemm;
    clblastSsyrk_t ssyrk;
    clblastDsyrk_t dsyrk;
    clblastStrsm_t strsm;
    clblastDtrsm_t dtrsm;
    // More function pointers to be added
} clblast_context_t;

/* ============================================================================
 * Lifecycle Management
 * ========================================================================== */

#ifdef _WIN32
    #include <windows.h>
    #define LOAD_LIBRARY(name) LoadLibraryA(name)
    #define GET_PROC_ADDRESS GetProcAddress
    #define CLOSE_LIBRARY FreeLibrary
    typedef HMODULE lib_handle_t;
#else
    #include <dlfcn.h>
    #define LOAD_LIBRARY(name) dlopen(name, RTLD_NOW)
    #define GET_PROC_ADDRESS dlsym
    #define CLOSE_LIBRARY dlclose
    typedef void* lib_handle_t;
#endif

/* DEPRECATED: Library loading is now handled by the plugin layer
 * The plugin calls fb_plugin_load_library() and passes the handle to trait.init()
 * This function is kept for reference but should not be called.
 */
#if 0
static void* load_clblast_library(void) {
#ifdef _WIN32
    const char* lib_names[] = {
        "clblast.dll",
        "C:\\libraries\\vcpkg\\installed\\x64-windows\\bin\\clblast.dll",
        "C:\\vcpkg\\installed\\x64-windows\\bin\\clblast.dll",
        NULL
    };
#elif defined(__APPLE__)
    const char* lib_names[] = {
        "libclblast.dylib",
        "/usr/local/lib/libclblast.dylib",
        "/opt/homebrew/lib/libclblast.dylib",
        NULL
    };
#else
    const char* lib_names[] = {
        "libclblast.so",
        "libclblast.so.1",
        "/usr/local/lib/libclblast.so",
        NULL
    };
#endif

    for (int i = 0; lib_names[i]; i++) {
        void* handle = LOAD_LIBRARY(lib_names[i]);
        if (handle) {
            return handle;
        }
    }
    return NULL;
}
#endif

static int clblast_init(int device_id, void* lib_handle, void** backend_handle) {
    cl_int err;
    
    // Find the OpenCL device corresponding to device_id
    // For now, simplified - assumes device_id maps directly to OpenCL enumeration
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "CLBlast: No OpenCL platforms found\n");
        return -1;
    }
    
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return -1;
    }
    
    // Find GPU device (simplified - should match device_id properly)
    cl_device_id target_device = NULL;
    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
            
            // TODO: Proper device_id mapping - for now take first GPU
            if (num_devices > 0) {
                target_device = devices[0];
                free(devices);
                break;
            }
            free(devices);
        }
    }
    free(platforms);
    
    if (!target_device) {
        fprintf(stderr, "CLBlast: No suitable GPU device found\n");
        return -1;
    }
    
    // Allocate context
    clblast_context_t* ctx = (clblast_context_t*)calloc(1, sizeof(clblast_context_t));
    if (!ctx) {
        return -1;
    }
    ctx->device = target_device;
    ctx->device_id = device_id;
    
    // Create OpenCL context
    ctx->context = clCreateContext(NULL, 1, &target_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "CLBlast: Failed to create OpenCL context: %d\n", err);
        free(ctx);
        return -1;
    }
    
    // Create command queue
    ctx->queue = clCreateCommandQueue(ctx->context, target_device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "CLBlast: Failed to create command queue: %d\n", err);
        clReleaseContext(ctx->context);
        free(ctx);
        return -1;
    }
    
    // Load CLBlast library and function pointers
    // Library handle provided by plugin - no need to load again
    ctx->lib_handle = lib_handle;
    if (!ctx->lib_handle) {
        fprintf(stderr, "CLBlast: Invalid library handle provided (NULL)\n");
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
        free(ctx);
        return -1;
    }
    
    // Load function pointers
    #define LOAD_FUNC(upper, lower) \
        ctx->lower = (clblast##upper##_t)GET_PROC_ADDRESS((lib_handle_t)ctx->lib_handle, "CLBlast" #upper); \
        if (!ctx->lower) { \
            fprintf(stderr, "CLBlast: Failed to load function CLBlast" #upper "\n"); \
        }
    
    LOAD_FUNC(Saxpy, saxpy);
    LOAD_FUNC(Daxpy, daxpy);
    LOAD_FUNC(Scopy, scopy);
    LOAD_FUNC(Dcopy, dcopy);
    LOAD_FUNC(Sscal, sscal);
    LOAD_FUNC(Dscal, dscal);
    LOAD_FUNC(Sdot, sdot);
    LOAD_FUNC(Ddot, ddot);
    LOAD_FUNC(Snrm2, snrm2);
    LOAD_FUNC(Dnrm2, dnrm2);
    LOAD_FUNC(Sgemv, sgemv);
    LOAD_FUNC(Dgemv, dgemv);
    LOAD_FUNC(Sger, sger);
    LOAD_FUNC(Dger, dger);
    LOAD_FUNC(Sgemm, sgemm);
    LOAD_FUNC(Dgemm, dgemm);
    LOAD_FUNC(Ssyrk, ssyrk);
    LOAD_FUNC(Dsyrk, dsyrk);
    LOAD_FUNC(Strsm, strsm);
    LOAD_FUNC(Dtrsm, dtrsm);
    
    #undef LOAD_FUNC
    
    *backend_handle = ctx;
    return 0;
}

static void clblast_shutdown(void* backend_handle) {
    if (!backend_handle) {
        return;
    }
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    if (ctx->queue) {
        clReleaseCommandQueue(ctx->queue);
    }
    if (ctx->context) {
        clReleaseContext(ctx->context);
    }
    if (ctx->lib_handle) {
        CLOSE_LIBRARY((lib_handle_t)ctx->lib_handle);
    }
    
    free(ctx);
}

static int clblast_get_device_properties(void* backend_handle, int device_id,
                                          char* name, size_t name_len,
                                          size_t* total_memory) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_int err;
    
    if (name && name_len > 0) {
        err = clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, name_len, name, NULL);
        if (err != CL_SUCCESS) {
            return -1;
        }
    }
    
    if (total_memory) {
        cl_ulong mem_size = 0;
        err = clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
        if (err != CL_SUCCESS) {
            return -1;
        }
        *total_memory = (size_t)mem_size;
    }
    
    return 0;
}

/* ============================================================================
 * Memory Management
 * ========================================================================== */

static int clblast_malloc(void* backend_handle, fb_gpu_ptr_t* ptr, size_t size) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_int err;
    
    cl_mem buffer = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "CLBlast: clCreateBuffer failed: %d\n", err);
        return -1;
    }
    
    *ptr = (fb_gpu_ptr_t)buffer;
    return 0;
}

static void clblast_free(void* backend_handle, fb_gpu_ptr_t ptr) {
    if (ptr) {
        clReleaseMemObject((cl_mem)ptr);
    }
}

static int clblast_memcpy_h2d(void* backend_handle, fb_gpu_ptr_t dst,
                               const void* src, size_t size) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_int err;
    
    err = clEnqueueWriteBuffer(ctx->queue, (cl_mem)dst, CL_TRUE, 0, size, src, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "CLBlast: clEnqueueWriteBuffer failed: %d\n", err);
        return -1;
    }
    
    return 0;
}

static int clblast_memcpy_d2h(void* backend_handle, void* dst,
                               fb_gpu_ptr_t src, size_t size) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_int err;
    
    err = clEnqueueReadBuffer(ctx->queue, (cl_mem)src, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "CLBlast: clEnqueueReadBuffer failed: %d\n", err);
        return -1;
    }
    
    return 0;
}

static int clblast_memcpy_d2d(void* backend_handle, fb_gpu_ptr_t dst,
                               fb_gpu_ptr_t src, size_t size) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_int err;
    
    err = clEnqueueCopyBuffer(ctx->queue, (cl_mem)src, (cl_mem)dst, 0, 0, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "CLBlast: clEnqueueCopyBuffer failed: %d\n", err);
        return -1;
    }
    
    return 0;
}

/* ============================================================================
 * Stream/Queue Management
 * ========================================================================== */

static int clblast_stream_create(void* backend_handle, fb_gpu_stream_t* stream) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_int err;
    
    cl_command_queue queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
    if (err != CL_SUCCESS) {
        return -1;
    }
    
    *stream = (fb_gpu_stream_t)queue;
    return 0;
}

static void clblast_stream_destroy(void* backend_handle, fb_gpu_stream_t stream) {
    if (stream) {
        clReleaseCommandQueue((cl_command_queue)stream);
    }
}

static int clblast_stream_synchronize(void* backend_handle, fb_gpu_stream_t stream) {
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    cl_int err = clFinish(queue);
    return (err == CL_SUCCESS) ? 0 : -1;
}

/* ============================================================================
 * Enum Conversion Helpers
 * ========================================================================== */

static int clblast_convert_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return CLBlastTransposeNo;
        case 'T': case 't': return CLBlastTransposeYes;
        case 'C': case 'c': return CLBlastTransposeConjugate;
        default: return CLBlastTransposeNo;
    }
}

static int clblast_convert_uplo(char uplo) {
    switch (uplo) {
        case 'U': case 'u': return CLBlastTriangleUpper;
        case 'L': case 'l': return CLBlastTriangleLower;
        default: return CLBlastTriangleUpper;
    }
}

static int clblast_convert_diag(char diag) {
    switch (diag) {
        case 'N': case 'n': return CLBlastDiagonalNonUnit;
        case 'U': case 'u': return CLBlastDiagonalUnit;
        default: return CLBlastDiagonalNonUnit;
    }
}

static int clblast_convert_side(char side) {
    switch (side) {
        case 'L': case 'l': return CLBlastSideLeft;
        case 'R': case 'r': return CLBlastSideRight;
        default: return CLBlastSideLeft;
    }
}

/* ============================================================================
 * BLAS Level 1 - Vector Operations
 * ========================================================================== */

static void clblast_saxpy_impl(void* handle, fb_gpu_stream_t stream,
                                int n, float alpha,
                                fb_gpu_ptr_t x, int incx,
                                fb_gpu_ptr_t y, int incy) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->saxpy) {
        ctx->saxpy(n, alpha, (cl_mem)x, 0, incx, (cl_mem)y, 0, incy, &queue, NULL);
    }
}

static void clblast_daxpy_impl(void* handle, fb_gpu_stream_t stream,
                                int n, double alpha,
                                fb_gpu_ptr_t x, int incx,
                                fb_gpu_ptr_t y, int incy) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->daxpy) {
        ctx->daxpy(n, alpha, (cl_mem)x, 0, incx, (cl_mem)y, 0, incy, &queue, NULL);
    }
}

static void clblast_scopy_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx,
                                fb_gpu_ptr_t y, int incy) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->scopy) {
        ctx->scopy(n, (cl_mem)x, 0, incx, (cl_mem)y, 0, incy, &queue, NULL);
    }
}

static void clblast_dcopy_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx,
                                fb_gpu_ptr_t y, int incy) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->dcopy) {
        ctx->dcopy(n, (cl_mem)x, 0, incx, (cl_mem)y, 0, incy, &queue, NULL);
    }
}

static void clblast_sscal_impl(void* handle, fb_gpu_stream_t stream,
                                int n, float alpha,
                                fb_gpu_ptr_t x, int incx) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->sscal) {
        ctx->sscal(n, alpha, (cl_mem)x, 0, incx, &queue, NULL);
    }
}

static void clblast_dscal_impl(void* handle, fb_gpu_stream_t stream,
                                int n, double alpha,
                                fb_gpu_ptr_t x, int incx) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->dscal) {
        ctx->dscal(n, alpha, (cl_mem)x, 0, incx, &queue, NULL);
    }
}

static void clblast_sdot_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy, float* result) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->sdot) {
        // Allocate temp buffer for result
        cl_mem result_buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
        ctx->sdot(n, result_buf, 0, (cl_mem)x, 0, incx, (cl_mem)y, 0, incy, &queue, NULL);
        clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, sizeof(float), result, 0, NULL, NULL);
        clReleaseMemObject(result_buf);
    }
}

static void clblast_ddot_impl(void* handle, fb_gpu_stream_t stream,
                               int n, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy, double* result) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->ddot) {
        // Allocate temp buffer for result
        cl_mem result_buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, sizeof(double), NULL, NULL);
        ctx->ddot(n, result_buf, 0, (cl_mem)x, 0, incx, (cl_mem)y, 0, incy, &queue, NULL);
        clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, sizeof(double), result, 0, NULL, NULL);
        clReleaseMemObject(result_buf);
    }
}

static void clblast_snrm2_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx, float* result) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->snrm2) {
        cl_mem result_buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
        ctx->snrm2(n, result_buf, 0, (cl_mem)x, 0, incx, &queue, NULL);
        clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, sizeof(float), result, 0, NULL, NULL);
        clReleaseMemObject(result_buf);
    }
}

static void clblast_dnrm2_impl(void* handle, fb_gpu_stream_t stream,
                                int n, fb_gpu_ptr_t x, int incx, double* result) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->dnrm2) {
        cl_mem result_buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, sizeof(double), NULL, NULL);
        ctx->dnrm2(n, result_buf, 0, (cl_mem)x, 0, incx, &queue, NULL);
        clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, sizeof(double), result, 0, NULL, NULL);
        clReleaseMemObject(result_buf);
    }
}


/* ============================================================================
 * BLAS Level 2 - Matrix-Vector Operations
 * ========================================================================== */

static void clblast_sgemv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                                int m, int n, float alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t x, int incx,
                                float beta,
                                fb_gpu_ptr_t y, int incy) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastTranspose cl_trans = (trans == 'N' || trans == 'n') ? 
                                 CLBlastTransposeNo : CLBlastTransposeYes;
    
    if (ctx->sgemv) {
        ctx->sgemv(CLBlastLayoutColMajor, cl_trans, m, n, alpha,
                   (cl_mem)A, 0, lda, (cl_mem)x, 0, incx,
                   beta, (cl_mem)y, 0, incy, &queue, NULL);
    }
}

static void clblast_dgemv_impl(void* handle, fb_gpu_stream_t stream, char trans,
                                int m, int n, double alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t x, int incx,
                                double beta,
                                fb_gpu_ptr_t y, int incy) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastTranspose cl_trans = (trans == 'N' || trans == 'n') ? 
                                 CLBlastTransposeNo : CLBlastTransposeYes;
    
    if (ctx->dgemv) {
        ctx->dgemv(CLBlastLayoutColMajor, cl_trans, m, n, alpha,
                   (cl_mem)A, 0, lda, (cl_mem)x, 0, incx,
                   beta, (cl_mem)y, 0, incy, &queue, NULL);
    }
}

static void clblast_sger_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, float alpha,
                               fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t A, int lda) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->sger) {
        ctx->sger(CLBlastLayoutColMajor, m, n, alpha,
                  (cl_mem)x, 0, incx, (cl_mem)y, 0, incy,
                  (cl_mem)A, 0, lda, &queue, NULL);
    }
}

static void clblast_dger_impl(void* handle, fb_gpu_stream_t stream,
                               int m, int n, double alpha,
                               fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy,
                               fb_gpu_ptr_t A, int lda) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    if (ctx->dger) {
        ctx->dger(CLBlastLayoutColMajor, m, n, alpha,
                  (cl_mem)x, 0, incx, (cl_mem)y, 0, incy,
                  (cl_mem)A, 0, lda, &queue, NULL);
    }
}

/* ============================================================================
 * BLAS Level 3 - Matrix-Matrix Operations
 * ========================================================================== */

static void clblast_sgemm_impl(void* handle, fb_gpu_stream_t stream,
                                char transa, char transb,
                                int m, int n, int k,
                                float alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb,
                                float beta,
                                fb_gpu_ptr_t C, int ldc) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastTranspose cl_transa = (transa == 'N' || transa == 'n') ? 
                                  CLBlastTransposeNo : CLBlastTransposeYes;
    CLBlastTranspose cl_transb = (transb == 'N' || transb == 'n') ? 
                                  CLBlastTransposeNo : CLBlastTransposeYes;
    
    if (ctx->sgemm) {
        ctx->sgemm(CLBlastLayoutColMajor, cl_transa, cl_transb,
                   m, n, k, alpha,
                   (cl_mem)A, 0, lda, (cl_mem)B, 0, ldb,
                   beta, (cl_mem)C, 0, ldc, &queue, NULL);
    }
}

static void clblast_dgemm_impl(void* handle, fb_gpu_stream_t stream,
                                char transa, char transb,
                                int m, int n, int k,
                                double alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb,
                                double beta,
                                fb_gpu_ptr_t C, int ldc) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastTranspose cl_transa = (transa == 'N' || transa == 'n') ? 
                                  CLBlastTransposeNo : CLBlastTransposeYes;
    CLBlastTranspose cl_transb = (transb == 'N' || transb == 'n') ? 
                                  CLBlastTransposeNo : CLBlastTransposeYes;
    
    if (ctx->dgemm) {
        ctx->dgemm(CLBlastLayoutColMajor, cl_transa, cl_transb,
                   m, n, k, alpha,
                   (cl_mem)A, 0, lda, (cl_mem)B, 0, ldb,
                   beta, (cl_mem)C, 0, ldc, &queue, NULL);
    }
}

static void clblast_ssyrk_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                float alpha, fb_gpu_ptr_t A, int lda,
                                float beta, fb_gpu_ptr_t C, int ldc) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastTriangle cl_uplo = (uplo == 'U' || uplo == 'u') ? 
                               CLBlastTriangleUpper : CLBlastTriangleLower;
    CLBlastTranspose cl_trans = (trans == 'N' || trans == 'n') ? 
                                 CLBlastTransposeNo : CLBlastTransposeYes;
    
    if (ctx->ssyrk) {
        ctx->ssyrk(CLBlastLayoutColMajor, cl_uplo, cl_trans, n, k, alpha,
                   (cl_mem)A, 0, lda, beta, (cl_mem)C, 0, ldc, &queue, NULL);
    }
}

static void clblast_dsyrk_impl(void* handle, fb_gpu_stream_t stream,
                                char uplo, char trans, int n, int k,
                                double alpha, fb_gpu_ptr_t A, int lda,
                                double beta, fb_gpu_ptr_t C, int ldc) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastTriangle cl_uplo = (uplo == 'U' || uplo == 'u') ? 
                               CLBlastTriangleUpper : CLBlastTriangleLower;
    CLBlastTranspose cl_trans = (trans == 'N' || trans == 'n') ? 
                                 CLBlastTransposeNo : CLBlastTransposeYes;
    
    if (ctx->dsyrk) {
        ctx->dsyrk(CLBlastLayoutColMajor, cl_uplo, cl_trans, n, k, alpha,
                   (cl_mem)A, 0, lda, beta, (cl_mem)C, 0, ldc, &queue, NULL);
    }
}

static void clblast_strsm_impl(void* handle, fb_gpu_stream_t stream,
                                char side, char uplo, char transa, char diag,
                                int m, int n, float alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastSide cl_side = (side == 'L' || side == 'l') ? 
                          CLBlastSideLeft : CLBlastSideRight;
    CLBlastTriangle cl_uplo = (uplo == 'U' || uplo == 'u') ? 
                               CLBlastTriangleUpper : CLBlastTriangleLower;
    CLBlastTranspose cl_transa = (transa == 'N' || transa == 'n') ? 
                                  CLBlastTransposeNo : CLBlastTransposeYes;
    CLBlastDiagonal cl_diag = (diag == 'N' || diag == 'n') ? 
                               CLBlastDiagonalNonUnit : CLBlastDiagonalUnit;
    
    if (ctx->strsm) {
        ctx->strsm(CLBlastLayoutColMajor, cl_side, cl_uplo, cl_transa, cl_diag,
                   m, n, alpha, (cl_mem)A, 0, lda, (cl_mem)B, 0, ldb, &queue, NULL);
    }
}

static void clblast_dtrsm_impl(void* handle, fb_gpu_stream_t stream,
                                char side, char uplo, char transa, char diag,
                                int m, int n, double alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb) {
    clblast_context_t* ctx = (clblast_context_t*)handle;
    cl_command_queue queue = stream ? (cl_command_queue)stream : ctx->queue;
    
    CLBlastSide cl_side = (side == 'L' || side == 'l') ? 
                          CLBlastSideLeft : CLBlastSideRight;
    CLBlastTriangle cl_uplo = (uplo == 'U' || uplo == 'u') ? 
                               CLBlastTriangleUpper : CLBlastTriangleLower;
    CLBlastTranspose cl_transa = (transa == 'N' || transa == 'n') ? 
                                  CLBlastTransposeNo : CLBlastTransposeYes;
    CLBlastDiagonal cl_diag = (diag == 'N' || diag == 'n') ? 
                               CLBlastDiagonalNonUnit : CLBlastDiagonalUnit;
    
    if (ctx->dtrsm) {
        ctx->dtrsm(CLBlastLayoutColMajor, cl_side, cl_uplo, cl_transa, cl_diag,
                   m, n, alpha, (cl_mem)A, 0, lda, (cl_mem)B, 0, ldb, &queue, NULL);
    }
}

// Stubs for remaining BLAS Level 3 functions
static void clblast_ssymm_impl(void* handle, fb_gpu_stream_t stream,
                                char side, char uplo, int m, int n,
                                const float* alpha, fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb, const float* beta,
                                fb_gpu_ptr_t C, int ldc) {
    // TODO: Implement
}

static void clblast_dsymm_impl(void* handle, fb_gpu_stream_t stream,
                                char side, char uplo, int m, int n,
                                const double* alpha, fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb, const double* beta,
                                fb_gpu_ptr_t C, int ldc) {
    // TODO: Implement
}

static void clblast_strmm_impl(void* handle, fb_gpu_stream_t stream,
                                char side, char uplo, char transa, char diag,
                                int m, int n, const float* alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb) {
    // TODO: Implement
}

static void clblast_dtrmm_impl(void* handle, fb_gpu_stream_t stream,
                                char side, char uplo, char transa, char diag,
                                int m, int n, const double* alpha,
                                fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t B, int ldb) {
    // TODO: Implement
}

/* ============================================================================
 * LAPACK Operations (Not supported by CLBlast - return error)
 * ========================================================================== */

static int clblast_sgetrf_impl(void* handle, fb_gpu_stream_t stream,
                                int m, int n, fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t ipiv) {
    fprintf(stderr, "CLBlast: LAPACK operations not supported\n");
    return -1;
}

static int clblast_dgetrf_impl(void* handle, fb_gpu_stream_t stream,
                                int m, int n, fb_gpu_ptr_t A, int lda,
                                fb_gpu_ptr_t ipiv) {
    fprintf(stderr, "CLBlast: LAPACK operations not supported\n");
    return -1;
}

/* ============================================================================
 * Trait Export
 * ========================================================================== */

const fb_gpu_backend_trait_t fb_clblast_trait = {
    .name = "clblast",
    .type = FB_GPU_BACKEND_CLBLAST,
    
    // Lifecycle
    .init = clblast_init,
    .shutdown = clblast_shutdown,
    .get_device_properties = clblast_get_device_properties,
    
    // Memory management
    .malloc = clblast_malloc,
    .free = clblast_free,
    .memcpy_h2d = clblast_memcpy_h2d,
    .memcpy_d2h = clblast_memcpy_d2h,
    .memcpy_d2d = clblast_memcpy_d2d,
    
    // Stream/queue management
    .stream_create = clblast_stream_create,
    .stream_destroy = clblast_stream_destroy,
    .stream_synchronize = clblast_stream_synchronize,
    
    // Enum conversion
    .convert_transpose = clblast_convert_transpose,
    .convert_uplo = clblast_convert_uplo,
    .convert_diag = clblast_convert_diag,
    .convert_side = clblast_convert_side,
    
    // BLAS Level 1
    .saxpy = clblast_saxpy_impl,
    .daxpy = clblast_daxpy_impl,
    .scopy = clblast_scopy_impl,
    .dcopy = clblast_dcopy_impl,
    .sscal = clblast_sscal_impl,
    .dscal = clblast_dscal_impl,
    .sdot = clblast_sdot_impl,
    .ddot = clblast_ddot_impl,
    .snrm2 = clblast_snrm2_impl,
    .dnrm2 = clblast_dnrm2_impl,
    .sasum = NULL,  // TODO
    .dasum = NULL,
    .isamax = NULL,
    .idamax = NULL,
    
    // BLAS Level 2
    .sgemv = clblast_sgemv_impl,
    .dgemv = clblast_dgemv_impl,
    .ssymv = NULL,  // TODO
    .dsymv = NULL,
    .strmv = NULL,
    .dtrmv = NULL,
    .sger = clblast_sger_impl,
    .dger = clblast_dger_impl,
    .ssyr = NULL,
    .dsyr = NULL,
    
    // BLAS Level 3
    .sgemm = clblast_sgemm_impl,
    .dgemm = clblast_dgemm_impl,
    .ssymm = clblast_ssymm_impl,
    .dsymm = clblast_dsymm_impl,
    .ssyrk = clblast_ssyrk_impl,
    .dsyrk = clblast_dsyrk_impl,
    .strmm = clblast_strmm_impl,
    .dtrmm = clblast_dtrmm_impl,
    .strsm = clblast_strsm_impl,
    .dtrsm = clblast_dtrsm_impl,
    
    // LAPACK - not supported
    .sgetrf = clblast_sgetrf_impl,
    .dgetrf = clblast_dgetrf_impl,
    .sgetrs = NULL,
    .dgetrs = NULL,
    .spotrf = NULL,
    .dpotrf = NULL,
    .spotrs = NULL,
    .dpotrs = NULL,
    .sgeqrf = NULL,
    .dgeqrf = NULL,
    .sgesvd = NULL,
    .dgesvd = NULL,
    .ssyev = NULL,
    .dsyev = NULL,
};
