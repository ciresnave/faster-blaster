# Backend Generation Guide

## Quick Start: Copy-Paste-Adapt Pattern

As you correctly observed, since BLAS/LAPACK operations have standardized signatures across implementations, we can rapidly generate new backends by adapting existing ones.

### Manual Method (5-10 minutes)

1. **Copy an existing backend file**:
   ```powershell
   cp src/backends/mkl_backend.c src/backends/blis_backend.c
   ```

2. **Find and replace** (in your editor):
   - `mkl` → `blis` (lowercase)
   - `MKL` → `BLIS` (uppercase)
   - `g_mkl` → `g_blis` (structure names)
   - `Mkl` → `Blis` (capitalized in comments)

3. **Update library names**:
   ```c
   // Windows
   static const char* windows_library[] = "libblis.dll";
   
   // Linux
   static const char* linux_library[] = "libblis.so";
   
   // macOS
   static const char* macos_library[] = "libblis.dylib";
   ```

4. **Verify and compile**

### Automated Method (1-2 minutes)

Use the provided script:

```powershell
.\tools\generate_backend.ps1 `
    -SourceBackend mkl `
    -TargetBackend blis `
    -SourceFile .\src\backends\mkl_backend.c `
    -OutputFile .\src\backends\blis_backend.c
```

## Backend Compatibility Matrix

| Backend        | CBLAS | LAPACKE | Notes                           |
| -------------- | ----- | ------- | ------------------------------- |
| **OpenBLAS**   | ✅     | ✅       | Reference implementation        |
| **Intel MKL**  | ✅     | ✅       | Complete, optimized             |
| **BLIS**       | ✅     | ✅       | Use libFLAME for LAPACK         |
| **AOCL**       | ✅     | ✅       | AMD optimized                   |
| **Accelerate** | ✅     | ✅       | macOS vecLib (no Windows/Linux) |
| **ATLAS**      | ✅     | ⚠️       | Limited LAPACK support          |
| **Netlib**     | ✅     | ✅       | Reference, not optimized        |

## What Gets Copied Automatically

✅ **No changes needed** (identical across backends):
- All Level 1 BLAS operations (54 functions)
- All Level 2 BLAS operations (70 functions)  
- All Level 3 BLAS operations (28 functions)
- All LAPACK operations (60 functions)
- Helper functions (`transpose_to_cblas`, `uplo_to_cblas`, etc.)
- FB_LOAD_SYMBOL macro
- Error handling patterns
- Vtable structure

## What Needs Manual Adjustment

⚠️ **Backend-specific changes**:
1. **Library loading paths**
   - Windows DLL name
   - Linux .so name  
   - macOS .dylib/.framework path

2. **Version detection** (optional)
   - Some backends expose version functions
   - MKL: `mkl_get_version_string()`
   - OpenBLAS: `openblas_get_config()`
   - BLIS: `bli_info_get_version_str()`

3. **Threading control** (optional)
   - MKL: `mkl_set_num_threads()`
   - OpenBLAS: `openblas_set_num_threads()`
   - BLIS: `bli_thread_set_num_threads()`

4. **Platform availability**
   - Accelerate: macOS only
   - AOCL: Optimized for AMD CPUs
   - MKL: Optimized for Intel CPUs

## Example: Generate BLIS Backend

### Step 1: Generate from MKL
```powershell
.\tools\generate_backend.ps1 -SourceBackend mkl -TargetBackend blis `
    -SourceFile .\src\backends\mkl_backend.c `
    -OutputFile .\src\backends\blis_backend.c
```

### Step 2: Adjust Threading (Optional)
```c
// In blis_backend.c, find the init function and add:
void fb_blis_set_num_threads(int num_threads) {
    typedef void (*fn_t)(int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "bli_thread_set_num_threads");
    if (fn) fn(num_threads);
}
```

### Step 3: Update Backend Loader
Add to `src/backends/backend_loader.c`:

```c
// In the priority array
static const fb_backend_info_t backend_priority[] = {
    {FB_BACKEND_MKL, "Intel MKL", fb_mkl_is_available, fb_mkl_get_vtable},
    {FB_BACKEND_OPENBLAS, "OpenBLAS", fb_openblas_is_available, fb_openblas_get_vtable},
    {FB_BACKEND_BLIS, "BLIS", fb_blis_is_available, fb_blis_get_vtable},  // NEW
    // ... rest
};
```

### Step 4: Compile and Test
```powershell
cmake --build build --target faster-blaster
.\build\Debug\faster-blaster.exe  # Should auto-detect BLIS if installed
```

## GPU Backends (Different Pattern)

⚠️ **GPU backends require different APIs** - cannot use simple copy-paste:

- **cuBLAS/cuSOLVER**: NVIDIA CUDA API (different signatures)
- **rocBLAS/rocSOLVER**: AMD ROCm API (similar to cuBLAS)

These need custom wrappers to translate:
```c
// CPU BLAS (CBLAS)
void cblas_sgemm(...);  // Synchronous, host memory

// GPU BLAS (cuBLAS)
cublasStatus_t cublasSgemm(cublasHandle_t handle, ...);  // Async, device memory
```

See `src/backends/gpu/cublas_backend.c` for the pattern.

## Time Savings

| Method                      | Time          | Lines Changed      |
| --------------------------- | ------------- | ------------------ |
| **Manual from scratch**     | 4-6 hours     | ~2000 lines        |
| **Manual copy-paste-adapt** | 10-15 minutes | ~50 changes        |
| **Automated script**        | 1-2 minutes   | 0 (script does it) |

## Validation Checklist

After generating a new backend:

- [ ] File compiles without errors
- [ ] Library loads on target platform
- [ ] `is_available()` returns true when library present
- [ ] Simple operation works (e.g., `saxpy`)
- [ ] Complex operation works (e.g., `dgesv`)
- [ ] Backend appears in `fb_get_current_backend_name()`
- [ ] Performance is reasonable for the hardware

## Common Issues

**Issue**: Library not found at runtime
- **Fix**: Check library search paths (`LD_LIBRARY_PATH`, `PATH`)

**Issue**: Symbol not found (e.g., `LAPACKE_dgesv`)
- **Fix**: Verify LAPACK is included in the library build

**Issue**: Crash on function call
- **Fix**: Check calling convention matches (layout parameter, enum conversions)

**Issue**: Wrong results
- **Fix**: Verify column-major layout (102) is used for LAPACKE calls

## Next Steps

1. ✅ Generate your first backend using the script
2. Test with a simple program
3. Add to CMake build system if needed
4. Update documentation
5. Submit PR or integrate into your project

The copy-paste-adapt pattern is the key to rapid backend expansion. You've identified exactly the right approach! 🎯
