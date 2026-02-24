# Faster-BLASTER API Reference

**Version**: 0.1.0  
**Last Updated**: December 14, 2025

## Overview

Faster-BLASTER provides a unified BLAS/LAPACK interface with automatic hardware-aware backend selection. The API follows standard CBLAS conventions while adding plugin management capabilities.

---

## Table of Contents

1. [Plugin System API](#plugin-system-api)
2. [BLAS Level 1 Operations](#blas-level-1-operations)
3. [BLAS Level 2 Operations](#blas-level-2-operations)
4. [BLAS Level 3 Operations](#blas-level-3-operations)
5. [Data Types](#data-types)
6. [Error Codes](#error-codes)

---

## Plugin System API

### Initialization

#### `fb_init_plugins()`

Initialize the plugin system and register all built-in plugins.

**Signature**:
```c
void fb_init_plugins(void);
```

**Parameters**: None

**Returns**: None

**Example**:
```c
#include "faster-blaster/backend_plugin.h"

int main(void) {
    fb_init_plugins();
    // Plugins are now registered and ready
    return 0;
}
```

**Notes**:
- Must be called before using `fb_load_best_plugin()` or `fb_enumerate_plugins()`
- Automatically registers all 9 built-in plugins (5 CPU + 4 GPU)
- Safe to call multiple times (subsequent calls are no-ops)

---

#### `fb_load_best_plugin()`

Probe all registered plugins and load the one with the highest compatibility score.

**Signature**:
```c
const fb_backend_plugin_t* fb_load_best_plugin(
    const char* backend_name,
    const char** search_paths,
    fb_plugin_context_t** ctx_out
);
```

**Parameters**:
- `backend_name` (const char*): Specific backend name to load, or `NULL` for automatic selection
- `search_paths` (const char**): NULL-terminated array of directory paths to search for libraries, or `NULL` for system defaults
- `ctx_out` (fb_plugin_context_t**): Output parameter for plugin context

**Returns**:
- `const fb_backend_plugin_t*`: Pointer to loaded plugin, or `NULL` on failure

**Example**:
```c
fb_plugin_context_t* ctx = NULL;
const fb_backend_plugin_t* plugin = fb_load_best_plugin(NULL, NULL, &ctx);

if (plugin) {
    printf("Loaded: %s v%s\n", 
           plugin->metadata->name, 
           plugin->metadata->version);
    
    // Use plugin...
    const fb_backend_vtable_t* vtable = plugin->get_vtable(ctx);
    
    // Clean up
    plugin->shutdown(ctx);
} else {
    fprintf(stderr, "Failed to load any plugin\n");
}
```

**Notes**:
- Automatically calls `probe()` on all plugins
- Selects plugin with highest score
- Calls `init()` on selected plugin
- `backend_name` can specify exact plugin (e.g., `"aocl-blis"`) to override automatic selection

---

#### `fb_enumerate_plugins()`

Iterate through all registered plugins.

**Signature**:
```c
size_t fb_enumerate_plugins(
    const fb_backend_plugin_t*** plugins_out
);
```

**Parameters**:
- `plugins_out` (const fb_backend_plugin_t***): Output array of plugin pointers

**Returns**:
- `size_t`: Number of registered plugins

**Example**:
```c
const fb_backend_plugin_t** plugins = NULL;
size_t count = fb_enumerate_plugins(&plugins);

printf("Registered Plugins: %zu\n", count);
for (size_t i = 0; i < count; i++) {
    const fb_plugin_metadata_t* meta = plugins[i]->metadata;
    printf("  %zu. %s v%s (%s)\n", 
           i + 1, meta->name, meta->version, meta->vendor);
}
```

---

### Plugin Registration

#### `fb_register_plugin()`

Register a custom plugin with the plugin system.

**Signature**:
```c
void fb_register_plugin(const fb_backend_plugin_t* plugin);
```

**Parameters**:
- `plugin` (const fb_backend_plugin_t*): Pointer to plugin structure

**Returns**: None

**Example**:
```c
static const fb_backend_plugin_t my_custom_plugin = {
    .metadata = &my_metadata,
    .probe = my_probe_func,
    .init = my_init_func,
    .get_vtable = my_get_vtable_func,
    .get_context = my_get_context_func,
    .shutdown = my_shutdown_func
};

void fb_register_my_custom_plugin(void) {
    fb_register_plugin(&my_custom_plugin);
}

// In your initialization code:
fb_init_plugins();
fb_register_my_custom_plugin();
```

---

### Library Loading Utilities

#### `fb_plugin_load_library()`

Cross-platform library loading helper.

**Signature**:
```c
fb_lib_handle_t fb_plugin_load_library(
    const char** lib_names,
    const char** search_paths
);
```

**Parameters**:
- `lib_names` (const char**): NULL-terminated array of library names to try
- `search_paths` (const char**): NULL-terminated array of directories to search

**Returns**:
- `fb_lib_handle_t`: Library handle, or `NULL` on failure

**Example**:
```c
const char* lib_names[] = {
#if defined(_WIN32)
    "mylib.dll",
#elif defined(__linux__)
    "libmylib.so",
#elif defined(__APPLE__)
    "libmylib.dylib",
#endif
    NULL
};

const char* search_paths[] = {
    "/usr/local/lib",
    "/opt/mylib/lib",
    NULL
};

fb_lib_handle_t handle = fb_plugin_load_library(lib_names, search_paths);
if (!handle) {
    fprintf(stderr, "Failed to load library\n");
}
```

**Platform Behavior**:
- **Windows**: Uses `LoadLibraryA()` / `LoadLibraryW()`
- **Linux**: Uses `dlopen()` with `RTLD_NOW | RTLD_LOCAL`
- **macOS**: Uses `dlopen()` with `RTLD_NOW | RTLD_LOCAL`

---

## Data Types

### Plugin Structures

#### `fb_plugin_metadata_t`

Plugin identification and capability information.

```c
typedef struct {
    const char* name;           // Plugin identifier (e.g., "aocl-blis")
    const char* version;        // Version string (e.g., "4.2.1")
    const char* vendor;         // Vendor name (e.g., "AMD")
    const char* description;    // Human-readable description
    uint32_t api_version;       // faster-blaster API version
    uint32_t capabilities;      // Capability flags (bitfield)
} fb_plugin_metadata_t;
```

**Capability Flags**:
```c
typedef enum {
    FB_PLUGIN_CAP_CPU          = 1 << 0,  // CPU execution
    FB_PLUGIN_CAP_GPU          = 1 << 1,  // GPU execution
    FB_PLUGIN_CAP_LEVEL1       = 1 << 2,  // BLAS Level 1 (vector ops)
    FB_PLUGIN_CAP_LEVEL2       = 1 << 3,  // BLAS Level 2 (matrix-vector ops)
    FB_PLUGIN_CAP_LEVEL3       = 1 << 4,  // BLAS Level 3 (matrix-matrix ops)
    FB_PLUGIN_CAP_LAPACK       = 1 << 5,  // LAPACK operations
    FB_PLUGIN_CAP_SINGLE_PREC  = 1 << 6,  // Single precision (float)
    FB_PLUGIN_CAP_DOUBLE_PREC  = 1 << 7,  // Double precision (double)
    FB_PLUGIN_CAP_COMPLEX      = 1 << 8,  // Complex types
    FB_PLUGIN_CAP_THREADSAFE   = 1 << 9,  // Thread-safe
} fb_plugin_capability_t;
```

**Example**:
```c
static const fb_plugin_metadata_t my_metadata = {
    .name = "my-backend",
    .version = "1.0.0",
    .vendor = "MyCompany",
    .description = "High-performance BLAS for XYZ hardware",
    .api_version = FB_API_VERSION_CURRENT,
    .capabilities = FB_PLUGIN_CAP_CPU | 
                   FB_PLUGIN_CAP_LEVEL1 | 
                   FB_PLUGIN_CAP_LEVEL2 | 
                   FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | 
                   FB_PLUGIN_CAP_DOUBLE_PREC |
                   FB_PLUGIN_CAP_THREADSAFE
};
```

---

#### `fb_plugin_probe_result_t`

Result of probing a plugin for compatibility.

```c
typedef struct {
    int score;                  // Compatibility score (0-100)
    const char* library_path;   // Path to backend library
    const char* reason;         // Human-readable explanation
} fb_plugin_probe_result_t;
```

**Score Interpretation**:
- **0**: Backend unavailable or incompatible
- **1-69**: Available but poor match
- **70-79**: Works, not optimized for this hardware
- **80-89**: Good portable option
- **90-95**: Vendor-optimized for this hardware
- **96-100**: Platform-exclusive, highly optimized

**Example**:
```c
fb_plugin_probe_result_t result = {
    .score = 95,
    .library_path = "/usr/lib/x86_64-linux-gnu/libblas.so.3",
    .reason = "Found AOCL BLIS on AMD CPU (optimal)"
};
```

---

#### `fb_backend_plugin_t`

Complete plugin interface definition.

```c
typedef struct {
    const fb_plugin_metadata_t* metadata;
    
    fb_plugin_probe_result_t (*probe)(
        fb_plugin_context_t* ctx, 
        const char** search_paths
    );
    
    int (*init)(
        fb_plugin_context_t* ctx, 
        const char* library_path
    );
    
    const fb_backend_vtable_t* (*get_vtable)(
        fb_plugin_context_t* ctx
    );
    
    void* (*get_context)(
        fb_plugin_context_t* ctx
    );
    
    void (*shutdown)(
        fb_plugin_context_t* ctx
    );
    
    void (*set_num_threads)(
        fb_plugin_context_t* ctx, 
        int num_threads
    );  // Optional
    
    int (*get_num_threads)(
        fb_plugin_context_t* ctx
    );  // Optional
} fb_backend_plugin_t;
```

---

#### `fb_backend_vtable_t`

Function pointer table for BLAS operations.

```c
typedef struct {
    /* BLAS Level 1 - Vector Operations */
    float (*sdot)(int n, const float* x, int incx, 
                  const float* y, int incy);
    double (*ddot)(int n, const double* x, int incx, 
                   const double* y, int incy);
    void (*saxpy)(int n, float alpha, const float* x, int incx, 
                  float* y, int incy);
    void (*daxpy)(int n, double alpha, const double* x, int incx, 
                  double* y, int incy);
    void (*scopy)(int n, const float* x, int incx, 
                  float* y, int incy);
    void (*dcopy)(int n, const double* x, int incx, 
                  double* y, int incy);
    void (*sscal)(int n, float alpha, float* x, int incx);
    void (*dscal)(int n, double alpha, double* x, int incx);
    float (*snrm2)(int n, const float* x, int incx);
    double (*dnrm2)(int n, const double* x, int incx);
    
    /* BLAS Level 2 - Matrix-Vector Operations */
    void (*sgemv)(
        int order, int trans, int m, int n, 
        float alpha, const float* A, int lda, 
        const float* x, int incx, 
        float beta, float* y, int incy
    );
    void (*dgemv)(
        int order, int trans, int m, int n, 
        double alpha, const double* A, int lda, 
        const double* x, int incx, 
        double beta, double* y, int incy
    );
    
    /* BLAS Level 3 - Matrix-Matrix Operations */
    void (*sgemm)(
        int order, int transA, int transB, 
        int m, int n, int k, 
        float alpha, const float* A, int lda, 
        const float* B, int ldb, 
        float beta, float* C, int ldc
    );
    void (*dgemm)(
        int order, int transA, int transB, 
        int m, int n, int k, 
        double alpha, const double* A, int lda, 
        const double* B, int ldb, 
        double beta, double* C, int ldc
    );
    
    /* ... (341 operations total) */
} fb_backend_vtable_t;
```

---

## BLAS Level 1 Operations

### `sdot` / `ddot` - Dot Product

Compute dot product of two vectors: $result = x \cdot y = \sum_{i=0}^{n-1} x_i \times y_i$

**Signature**:
```c
float fb_sdot(int n, const float* x, int incx, const float* y, int incy);
double fb_ddot(int n, const double* x, int incx, const double* y, int incy);
```

**Parameters**:
- `n`: Number of elements
- `x`: First vector
- `incx`: Stride of x (1 for contiguous)
- `y`: Second vector
- `incy`: Stride of y (1 for contiguous)

**Returns**: Dot product result

**Example**:
```c
float x[] = {1.0f, 2.0f, 3.0f};
float y[] = {4.0f, 5.0f, 6.0f};
float result = fb_sdot(3, x, 1, y, 1);
// result = 1*4 + 2*5 + 3*6 = 32.0
```

**Performance**: O(n)

---

### `saxpy` / `daxpy` - Scalar Alpha X Plus Y

Compute $y = \alpha x + y$

**Signature**:
```c
void fb_saxpy(int n, float alpha, const float* x, int incx, 
              float* y, int incy);
void fb_daxpy(int n, double alpha, const double* x, int incx, 
              double* y, int incy);
```

**Parameters**:
- `n`: Number of elements
- `alpha`: Scalar multiplier
- `x`: Source vector (read-only)
- `incx`: Stride of x
- `y`: Destination vector (read-write, updated in-place)
- `incy`: Stride of y

**Example**:
```c
float x[] = {1.0f, 2.0f, 3.0f};
float y[] = {4.0f, 5.0f, 6.0f};
fb_saxpy(3, 2.0f, x, 1, y, 1);
// y = 2*x + y = {2*1+4, 2*2+5, 2*3+6} = {6.0, 9.0, 12.0}
```

**Performance**: O(n)

---

### `scopy` / `dcopy` - Vector Copy

Copy vector x to y: $y = x$

**Signature**:
```c
void fb_scopy(int n, const float* x, int incx, float* y, int incy);
void fb_dcopy(int n, const double* x, int incx, double* y, int incy);
```

**Parameters**:
- `n`: Number of elements
- `x`: Source vector
- `incx`: Stride of x
- `y`: Destination vector
- `incy`: Stride of y

**Example**:
```c
float x[] = {1.0f, 2.0f, 3.0f};
float y[3];
fb_scopy(3, x, 1, y, 1);
// y = {1.0, 2.0, 3.0}
```

**Performance**: O(n)

---

### `sscal` / `dscal` - Vector Scaling

Scale vector by scalar: $x = \alpha x$

**Signature**:
```c
void fb_sscal(int n, float alpha, float* x, int incx);
void fb_dscal(int n, double alpha, double* x, int incx);
```

**Parameters**:
- `n`: Number of elements
- `alpha`: Scalar multiplier
- `x`: Vector (updated in-place)
- `incx`: Stride of x

**Example**:
```c
float x[] = {1.0f, 2.0f, 3.0f};
fb_sscal(3, 2.0f, x, 1);
// x = 2*x = {2.0, 4.0, 6.0}
```

**Performance**: O(n)

---

### `snrm2` / `dnrm2` - Euclidean Norm

Compute Euclidean norm: $\|x\|_2 = \sqrt{\sum_{i=0}^{n-1} x_i^2}$

**Signature**:
```c
float fb_snrm2(int n, const float* x, int incx);
double fb_dnrm2(int n, const double* x, int incx);
```

**Parameters**:
- `n`: Number of elements
- `x`: Vector
- `incx`: Stride of x

**Returns**: Euclidean norm

**Example**:
```c
float x[] = {3.0f, 4.0f};
float norm = fb_snrm2(2, x, 1);
// norm = sqrt(3^2 + 4^2) = sqrt(25) = 5.0
```

**Performance**: O(n)

---

## BLAS Level 2 Operations

### `sgemv` / `dgemv` - General Matrix-Vector Multiply

Compute $y = \alpha \cdot op(A) \cdot x + \beta \cdot y$

where $op(A) = A$ or $op(A) = A^T$

**Signature**:
```c
void fb_sgemv(
    int order, int trans, int m, int n, 
    float alpha, const float* A, int lda, 
    const float* x, int incx, 
    float beta, float* y, int incy
);
```

**Parameters**:
- `order`: `CblasRowMajor` (101) or `CblasColMajor` (102)
- `trans`: `CblasNoTrans` (111), `CblasTrans` (112), or `CblasConjTrans` (113)
- `m`: Number of rows in matrix A
- `n`: Number of columns in matrix A
- `alpha`: Scalar multiplier for A*x
- `A`: Matrix (m × n)
- `lda`: Leading dimension of A
- `x`: Vector (length n if NoTrans, m if Trans)
- `incx`: Stride of x
- `beta`: Scalar multiplier for y
- `y`: Result vector (length m if NoTrans, n if Trans)
- `incy`: Stride of y

**Example**:
```c
// Compute y = 2*A*x + 3*y
// A = [1 2]    x = [5]    y = [7]
//     [3 4]        [6]        [8]

float A[] = {1, 2, 3, 4};  // Row-major
float x[] = {5, 6};
float y[] = {7, 8};

fb_sgemv(CblasRowMajor, CblasNoTrans, 2, 2, 
         2.0f, A, 2, x, 1, 3.0f, y, 1);

// Result: y = 2*[1*5+2*6, 3*5+4*6] + 3*[7, 8]
//           = 2*[17, 39] + [21, 24]
//           = [34, 78] + [21, 24]
//           = [55, 102]
```

**Performance**: O(m × n)

---

## BLAS Level 3 Operations

### `sgemm` / `dgemm` - General Matrix-Matrix Multiply

Compute $C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C$

where $op(X) = X$ or $op(X) = X^T$

**Signature**:
```c
void fb_sgemm(
    int order, int transA, int transB, 
    int m, int n, int k, 
    float alpha, const float* A, int lda, 
    const float* B, int ldb, 
    float beta, float* C, int ldc
);
```

**Parameters**:
- `order`: `CblasRowMajor` or `CblasColMajor`
- `transA`: Transpose mode for A
- `transB`: Transpose mode for B
- `m`: Number of rows in op(A) and C
- `n`: Number of columns in op(B) and C
- `k`: Number of columns in op(A) and rows in op(B)
- `alpha`: Scalar multiplier for A*B
- `A`: First matrix (m × k if NoTrans, k × m if Trans)
- `lda`: Leading dimension of A
- `B`: Second matrix (k × n if NoTrans, n × k if Trans)
- `ldb`: Leading dimension of B
- `C`: Result matrix (m × n), updated in-place
- `ldc`: Leading dimension of C
- `beta`: Scalar multiplier for C

**Example**:
```c
// Compute C = A*B
// A = [1 2]    B = [5 6]    C = [0 0]
//     [3 4]        [7 8]        [0 0]

float A[] = {1, 2, 3, 4};
float B[] = {5, 6, 7, 8};
float C[] = {0, 0, 0, 0};

fb_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
         2, 2, 2, 
         1.0f, A, 2, B, 2, 0.0f, C, 2);

// Result: C = [1*5+2*7  1*6+2*8]   [19 22]
//             [3*5+4*7  3*6+4*8] = [43 50]
```

**Performance**: O(m × n × k)

**Note**: GEMM is the most computationally intensive BLAS operation and benefits greatly from hardware acceleration (GPU, vendor-optimized libraries).

---

## Error Codes

```c
#define FB_SUCCESS           0   // Operation succeeded
#define FB_ERROR_INIT       -1   // Initialization failed
#define FB_ERROR_LOAD       -2   // Library loading failed
#define FB_ERROR_SYMBOL     -3   // Required symbol not found
#define FB_ERROR_INVALID    -4   // Invalid parameter
#define FB_ERROR_UNSUPPORTED -5  // Operation not supported
```

---

## Complete Example

```c
#include "faster-blaster/backend_plugin.h"
#include <stdio.h>

int main(void) {
    // Initialize plugin system
    fb_init_plugins();
    
    // Load best plugin for this hardware
    fb_plugin_context_t* ctx = NULL;
    const fb_backend_plugin_t* plugin = fb_load_best_plugin(NULL, NULL, &ctx);
    
    if (!plugin) {
        fprintf(stderr, "Failed to load any backend\n");
        return 1;
    }
    
    printf("Loaded: %s v%s (%s)\n", 
           plugin->metadata->name,
           plugin->metadata->version,
           plugin->metadata->vendor);
    
    // Get function table
    const fb_backend_vtable_t* vtable = plugin->get_vtable(ctx);
    
    // Perform dot product
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    float result = vtable->sdot(3, x, 1, y, 1);
    
    printf("Dot product: %.1f\n", result);  // Output: 32.0
    
    // Perform matrix multiplication: C = A * B
    float A[] = {1, 2, 3, 4};  // 2×2 matrix
    float B[] = {5, 6, 7, 8};  // 2×2 matrix
    float C[] = {0, 0, 0, 0};  // 2×2 result
    
    vtable->sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        2, 2, 2,
        1.0f, A, 2, B, 2, 0.0f, C, 2
    );
    
    printf("Matrix product:\n");
    printf("  [%.0f %.0f]\n", C[0], C[1]);
    printf("  [%.0f %.0f]\n", C[2], C[3]);
    // Output:
    //   [19 22]
    //   [43 50]
    
    // Cleanup
    plugin->shutdown(ctx);
    
    return 0;
}
```

**Compile**:
```bash
gcc example.c -I/usr/local/include -L/usr/local/lib -lfaster-blaster -o example
```

**Run**:
```bash
./example
```

---

## See Also

- [PLUGIN_ARCHITECTURE.md](PLUGIN_ARCHITECTURE.md) - Plugin system internals
- [BUILD.md](BUILD.md) - Build instructions
- [Standard CBLAS Reference](https://www.netlib.org/blas/)
- [LAPACK Reference](https://www.netlib.org/lapack/)

---

*Last updated: December 14, 2025*
