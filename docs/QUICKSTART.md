# Quick Start Guide - Using Multiple Backends

## Installation

### Step 1: Build faster-blaster

```powershell
# Windows PowerShell
cd C:\Users\cires\OneDrive\Documents\projects\faster-blaster
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Step 2: Test Backend Detection

```powershell
# Run the backend test
.\Release\backend_test.exe
```

**Expected Output:**
```
Faster-Blaster Backend Test
========================================

Initializing backend loader...

=== Available Backends ===
Found 2 backend(s):

  [0] Reference
      Version:      1.0.0
      License:      MIT
      Vendor:       Faster-Blaster
      Priority:     1
      Capabilities: CPU Level1 Level2 Level3 LAPACK 

  [1] OpenBLAS
      Version:      0.3.x
      License:      BSD-3-Clause
      Vendor:       OpenBLAS Project
      Priority:     50
      Capabilities: CPU Level1 Level2 Level3 LAPACK ThreadSafe 

=== Auto-selecting Backend ===
Selected: OpenBLAS (priority 50)

=== Testing Basic BLAS Operations ===
SDOT result: 220.00
SAXPY completed (alpha=2.0)
SNRM2 result: 577.35

Tip: Run with --benchmark flag to compare backend performance
```

### Step 3: Run Performance Benchmark

```powershell
.\Release\backend_test.exe --benchmark
```

**Example Output:**
```
=== DGEMM Benchmark (size=1000, iterations=10) ===
Reference            :    12.34 GFLOPS (1.622 sec)
OpenBLAS             :   145.67 GFLOPS (0.137 sec)
```

## Usage in Your Code

### Basic Example - Auto-Select Backend

```c
#include "backend_loader.h"
#include <stdio.h>

int main(void) {
    // 1. Initialize backend system
    fb_backend_loader_init();
    
    // 2. Auto-select best available backend
    fb_backend_type_t backend = fb_backend_auto_select(false);
    fb_backend_set_current(backend);
    
    // 3. Get backend info
    fb_backend_info_t info;
    fb_backend_get_info(backend, &info);
    printf("Using: %s\n", info.name);
    
    // 4. Load backend vtable
    fb_backend_vtable_t vtable;
    if (fb_backend_load(backend, &vtable) == 0) {
        // 5. Use operations
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float result;
        
        vtable.sdot(4, x, 1, y, 1, &result);
        printf("Dot product: %f\n", result);  // 70.0
    }
    
    // 6. Cleanup
    fb_backend_loader_shutdown();
    return 0;
}
```

### Advanced Example - Explicit Backend Selection

```c
#include "backend_loader.h"
#include <stdio.h>

int main(void) {
    fb_backend_loader_init();
    
    // Try to use Intel MKL if available
    if (fb_backend_set_current(FB_BACKEND_MKL) == 0) {
        printf("Using Intel MKL\n");
        
        // Configure MKL threading
        extern void fb_mkl_set_num_threads(int);
        fb_mkl_set_num_threads(8);
        
    } else {
        // Fallback to OpenBLAS
        printf("MKL not available, using OpenBLAS\n");
        fb_backend_set_current(FB_BACKEND_OPENBLAS);
    }
    
    // Use operations...
    
    fb_backend_loader_shutdown();
    return 0;
}
```

### GPU Example

```c
#include "backend_loader.h"
#include "gpu/gpu_backend.h"
#include "gpu/cublas_backend.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    fb_backend_loader_init();
    
    // Check CUDA availability
    if (!fb_cublas_is_available()) {
        printf("CUDA not available\n");
        return 1;
    }
    
    printf("CUDA devices: %d\n", fb_cublas_device_count());
    
    // Initialize GPU
    fb_gpu_context_t ctx;
    fb_cublas_init(0, &ctx);  // Device 0
    
    // Allocate vectors
    const int N = 1000000;
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(N - i);
    }
    
    // Get GPU backend
    const fb_gpu_backend_t* gpu = fb_cublas_get_backend();
    
    // Allocate device memory
    fb_gpu_ptr_t d_x, d_y;
    gpu->malloc(&d_x, N * sizeof(float));
    gpu->malloc(&d_y, N * sizeof(float));
    
    // Transfer to GPU
    gpu->memcpy_h2d(d_x, h_x, N * sizeof(float), NULL);
    gpu->memcpy_h2d(d_y, h_y, N * sizeof(float), NULL);
    
    printf("Data on GPU, ready for operations\n");
    
    // TODO: Perform GPU BLAS operations when vtable is complete
    
    // Cleanup
    gpu->free(d_x);
    gpu->free(d_y);
    fb_cublas_shutdown(&ctx);
    
    free(h_x);
    free(h_y);
    
    fb_backend_loader_shutdown();
    return 0;
}
```

## Common Patterns

### Pattern 1: Try Multiple Backends

```c
fb_backend_type_t try_order[] = {
    FB_BACKEND_MKL,
    FB_BACKEND_OPENBLAS,
    FB_BACKEND_REFERENCE
};

for (int i = 0; i < 3; i++) {
    if (fb_backend_set_current(try_order[i]) == 0) {
        fb_backend_info_t info;
        fb_backend_get_info(try_order[i], &info);
        printf("Using: %s\n", info.name);
        break;
    }
}
```

### Pattern 2: List All Backends

```c
fb_backend_info_t backends[FB_BACKEND_COUNT];
int count = fb_backend_list_available(backends, FB_BACKEND_COUNT);

printf("Available backends:\n");
for (int i = 0; i < count; i++) {
    printf("  %s (priority: %d)\n", backends[i].name, backends[i].priority);
}
```

### Pattern 3: Prefer GPU

```c
// Auto-select with GPU preference
fb_backend_type_t backend = fb_backend_auto_select(true);

fb_backend_info_t info;
fb_backend_get_info(backend, &info);

if (info.capabilities & FB_CAP_GPU) {
    printf("Using GPU backend: %s\n", info.name);
} else {
    printf("No GPU available, using CPU: %s\n", info.name);
}
```

### Pattern 4: Error Handling

```c
fb_backend_vtable_t vtable;
if (fb_backend_load(FB_BACKEND_MKL, &vtable) < 0) {
    fprintf(stderr, "Error: %s\n", fb_backend_get_error());
    // Try fallback
    fb_backend_load(FB_BACKEND_REFERENCE, &vtable);
}
```

## Installing Optional Backends

### Intel MKL (Windows)

1. Download Intel oneAPI Base Toolkit
2. Run installer
3. Add to PATH:
```powershell
$env:PATH += ";C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64"
```
4. Restart program - MKL will be auto-detected

### Intel MKL (Linux)

```bash
# Install
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/...
sudo sh l_BaseKit_...sh

# Setup environment
source /opt/intel/oneapi/setvars.sh

# Verify
./backend_test
```

### OpenBLAS (If Not Bundled)

**Windows:**
```powershell
# Download from GitHub releases
# Place libopenblas.dll in PATH or app directory
```

**Linux:**
```bash
sudo apt install libopenblas-dev  # Ubuntu/Debian
sudo dnf install openblas-devel   # Fedora
```

**macOS:**
```bash
brew install openblas
```

### NVIDIA CUDA (for cuBLAS)

1. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install (follow wizard)
3. Verify:
```powershell
nvidia-smi
nvcc --version
```
4. Run GPU test:
```powershell
.\Release\gpu_example.exe
```

## Performance Tips

### 1. Thread Configuration

```c
// For OpenBLAS
fb_openblas_set_num_threads(8);

// For MKL
fb_mkl_set_num_threads(8);
fb_mkl_set_threading_layer("intel");  // or "tbb", "gnu"
```

### 2. CPU Affinity (Linux)

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

### 3. Large Matrix Optimization

```c
// For matrices > 1000x1000, strongly prefer MKL or GPU
if (matrix_size > 1000) {
    fb_backend_type_t backend = fb_backend_auto_select(true);  // Prefer GPU
}
```

### 4. Benchmark Your Workload

```c
#include <time.h>

clock_t start = clock();
// ... perform operations ...
clock_t end = clock();

double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
printf("Time: %.3f seconds\n", elapsed);
```

## Troubleshooting

### Problem: "Backend not available"

**Solution:** Check library is in PATH/LD_LIBRARY_PATH

```powershell
# Windows - check PATH
$env:PATH

# Add directory
$env:PATH += ";C:\path\to\library"
```

```bash
# Linux - check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Add directory
export LD_LIBRARY_PATH=/path/to/library:$LD_LIBRARY_PATH
```

### Problem: "Operation returned NULL"

**Solution:** Operation not implemented in backend

```c
fb_backend_vtable_t vtable;
fb_backend_load(backend, &vtable);

if (!vtable.sgemm) {
    printf("SGEMM not available, use different backend\n");
}
```

### Problem: Poor Performance

**Solutions:**
1. Check thread count: `fb_mkl_get_num_threads()`
2. Use MKL instead of OpenBLAS for Intel CPUs
3. Use GPU backend for large problems (N > 1000)
4. Enable CPU affinity (Linux)

### Problem: GPU Operations Slow

**Solutions:**
1. Ensure data is already on GPU (avoid transfers)
2. Use async operations with streams
3. Batch operations when possible
4. Check GPU isn't shared with display

## Integration with CMake

### In Your Project's CMakeLists.txt

```cmake
# Find faster-blaster
find_package(faster_blaster REQUIRED)

# Link your executable
add_executable(myapp main.c)
target_link_libraries(myapp faster_blaster::faster_blaster)
```

### Or use as Subdirectory

```cmake
add_subdirectory(faster-blaster)

add_executable(myapp main.c)
target_link_libraries(myapp faster_blaster)
```

## Next Steps

1. ✅ Run `backend_test.exe` to verify installation
2. ✅ Try basic example with auto-selection
3. ✅ Install optional backends (MKL, CUDA)
4. ✅ Run benchmarks to compare performance
5. ✅ Integrate into your project
6. 📖 Read `docs/backend_plugin_architecture.md` for advanced usage
7. 📖 Read `docs/custom_backend_example.md` to create custom backends

## Support & Resources

- **Architecture:** `docs/backend_plugin_architecture.md`
- **Custom Backends:** `docs/custom_backend_example.md`
- **Build Guide:** `docs/BUILD.md`
- **Examples:** `examples/backend_test.c`, `examples/gpu_example.c`
- **Implementation Status:** `docs/BACKEND_IMPLEMENTATION_SUMMARY.md`

---

**You now have a working multi-backend BLAS/LAPACK system!** 🎉

Start with the Reference backend (always works), then add OpenBLAS for better performance, MKL for Intel CPUs, and cuBLAS/rocBLAS for GPU acceleration as needed.
