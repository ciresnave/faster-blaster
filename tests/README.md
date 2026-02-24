# Faster-Blaster Backend Tests

This directory contains comprehensive tests for all faster-blaster backend implementations, covering both GPU and CPU backends.

## Test Files

### GPU Backend Tests

- **test_cublas_basic.c** - NVIDIA cuBLAS GPU backend (CUDA)
  - Lifecycle management (init/shutdown)
  - Device properties query
  - Memory management (malloc/free/memcpy)
  - Stream operations
  - BLAS Level 1, 2, 3 operations

- **test_onemkl_backend.cpp** - Intel oneMKL GPU backend (SYCL/CUDA)
  - Device initialization and properties
  - Memory allocation and transfers
  - SAXPY, SGEMV, SGEMM operations
  - Large matrix multiplication (512x512)
  - Works on Intel GPUs or NVIDIA GPUs via CUDA backend

### CPU Backend Tests

- **test_aocl_backend.c** - AMD AOCL CPU backend
  - Optimized for AMD Ryzen/EPYC processors
  - BLAS Level 1 (SAXPY, SDOT)
  - BLAS Level 2 (SGEMV)
  - BLAS Level 3 (SGEMM, DGEMM)
  - Threading configuration
  - Large matrix performance tests (512x512)

- **test_blis_backend.c** - BLIS portable CPU backend
  - Portable BLAS implementation
  - Works on x86, ARM, RISC-V
  - BLAS Level 1 (SAXPY, SDOT, SNRM2)
  - BLAS Level 2 (SGEMV)
  - BLAS Level 3 (SGEMM, DGEMM)
  - Multi-threading support

### Reference & Correctness Tests

- **test_blas_level1_reference_expanded.c** - Full 54-operation BLAS Level 1 reference suite
  - Standalone, inline implementations (no external BLAS linkage)
  - Validates all S/D/C/Z variants for Level 1
  - Intended as a correctness oracle for tiny-size buckets

- **test_correctness_with_reference.c** - Cross-backend correctness validation
  - Compares backend outputs against the reference backend
  - Emits JSON results for CI/CD integration

- **test_lapack_cpu_reference.c** - CPU LAPACK reference tests
  - Baseline correctness for LAPACK kernels

## Building Tests

### Prerequisites

**For GPU Tests:**
- NVIDIA CUDA Toolkit 11.0+ (cuBLAS)
- Intel oneAPI Base Toolkit 2024.0+ (oneMKL)
- NVIDIA GPU with compute capability 3.5+

**For CPU Tests:**
- AMD AOCL library (for AOCL tests)
- BLIS library (for BLIS tests)
- Windows 10/11 or Linux

### Build Instructions

#### Using CMake (Recommended)

```powershell
# Configure with desired backends
cmake -B build -DCMAKE_BUILD_TYPE=Release `
    -DENABLE_CUBLAS=ON `
    -DENABLE_ONEMKL=ON `
    -DENABLE_AOCL=ON `
    -DENABLE_BLIS=ON

# Build all tests
cmake --build build --config Release

# Or build specific test
cmake --build build --config Release --target test_cublas_basic
cmake --build build --config Release --target test_aocl_backend
cmake --build build --config Release --target test_blis_backend
cmake --build build --config Release --target test_onemkl_backend
```

#### Using PowerShell Script

The **test_all_backends.ps1** script builds and runs all enabled backend tests:

```powershell
# Run all tests
cd tests
.\test_all_backends.ps1

# Run specific backends only
.\test_all_backends.ps1 -Backends cublas,aocl

# Use Debug build
.\test_all_backends.ps1 -Config Debug
```

## Running Tests

### Individual Tests

After building, run tests from the build directory:

```powershell
# cuBLAS tests
.\build\tests\Release\test_cublas_basic.exe

# AOCL tests
.\build\tests\Release\test_aocl_backend.exe

# BLIS tests
.\build\tests\Release\test_blis_backend.exe

# oneMKL tests
.\build\tests\Release\test_onemkl_backend.exe

# Reference BLAS Level 1 expanded tests
.\build\tests\Release\test_blas_level1_reference_expanded.exe
```

### All Tests at Once

```powershell
cd tests
.\test_all_backends.ps1
```

### Using CTest

```powershell
cd build
ctest -C Release --verbose
```

## Expected Output

### Successful Test Output

```
╔═══════════════════════════════════════════════════════════╗
║       AMD AOCL Backend Test Suite                         ║
║       Testing on: AMD Ryzen 9 7950X (expected)           ║
╚═══════════════════════════════════════════════════════════╝

📋 Test 1: AOCL Availability
═══════════════════════════════════════
  AOCL library available: Yes
  AOCL version: 4.2.0
  ✅ PASSED AOCL availability check

📋 Test 2: AOCL Initialization
═══════════════════════════════════════
  Initialization result: 0
  Default threads: 16
  After setting to 8: 8
  After auto: 16
  ✅ PASSED AOCL initialization

...

╔═══════════════════════════════════════════════════════════╗
║                   Test Summary                            ║
╠═══════════════════════════════════════════════════════════╣
║  Total Tests: 8                                           ║
║  Passed: 8                                                ║
║  Failed: 0                                                ║
╚═══════════════════════════════════════════════════════════╝
```

## Test Coverage

### Each backend test validates:

✅ **Initialization & Shutdown**
- Backend library loading
- Version detection
- Resource initialization
- Cleanup on shutdown

✅ **Configuration**
- Thread count setting
- Device selection
- Memory limits

✅ **BLAS Level 1** (Vector operations)
- SAXPY: y = α*x + y
- SDOT: dot product
- SNRM2: Euclidean norm
- SCOPY, SSCAL, SSWAP

✅ **BLAS Level 2** (Matrix-vector operations)
- SGEMV: y = α*A*x + β*y
- DGEMV: Double precision version

✅ **BLAS Level 3** (Matrix-matrix operations)
- SGEMM: C = α*A*B + β*C
- DGEMM: Double precision version
- Large matrix multiplication (performance test)

✅ **Numerical Correctness**
- Result verification against expected values
- Tolerance checking (1e-5 single, 1e-12 double)
- NaN/Inf detection

## Troubleshooting

### AOCL Tests Fail

```
⚠️  AOCL not found on system. Install AMD AOCL from:
    https://developer.amd.com/amd-aocl/
```

**Solution**: Download and install AMD AOCL library, ensure it's in your PATH.

### BLIS Tests Fail

```
⚠️  BLIS not found on system
```

**Solution**:
- Windows: `vcpkg install blis`
- Linux: `apt install libblis-dev` or `yum install blis-devel`

### oneMKL Tests Fail

```
oneMKL: No GPU devices found
```

**Solution**: Install Intel oneAPI Base Toolkit and configure oneMKL CUDA backend for NVIDIA GPUs.

### Build Fails

If CMake can't find backend libraries:

```powershell
# Disable problematic backends
cmake -B build -DENABLE_AOCL=OFF -DENABLE_BLIS=OFF -DENABLE_ONEMKL=OFF
```

## Hardware Requirements

### Tested Configurations

| Backend | Hardware               | OS         | Status             |
| ------- | ---------------------- | ---------- | ------------------ |
| cuBLAS  | NVIDIA RTX 4090        | Windows 11 | ✅ Production Ready |
| AOCL    | AMD Ryzen 9 7950X      | Windows 11 | 🧪 Needs Testing    |
| BLIS    | AMD Ryzen 9 7950X      | Windows 11 | 🧪 Needs Testing    |
| oneMKL  | Intel Arc A770         | Windows 11 | ⏸️ Requires oneAPI  |
| oneMKL  | NVIDIA RTX 4090 (CUDA) | Windows 11 | 🧪 Needs Testing    |

## Contributing

When adding new backend tests:

1. Create `test_<backend>_backend.c` (or .cpp)
2. Implement at least 7 test functions covering:
   - Availability check
   - Initialization
   - Level 1, 2, 3 BLAS operations
   - Large matrix test
   - Numerical verification
3. Add build rules to `CMakeLists.txt`
4. Update `test_all_backends.ps1` with new backend
5. Update this README

## License

Copyright (c) 2025  
Licensed under MIT OR Apache-2.0
.\test_cublas_basic.exe
```

### Expected Output

```
==================================================
cuBLAS Backend Trait - Basic Functionality Tests
==================================================

=== Testing Lifecycle ===
PASS: cuBLAS initialization
PASS: Handle is not NULL
PASS: Shutdown completed

=== Testing Device Properties ===
PASS: Get device properties
Device: NVIDIA GeForce RTX 4090
Memory: 24.00 GB

=== Testing Memory Management ===
PASS: Device memory allocation
PASS: Host to device transfer
PASS: Device to host transfer
PASS: Data integrity after H2D and D2H
PASS: Device to device transfer

=== Testing Stream Operations ===
PASS: Stream creation
PASS: Stream synchronization
PASS: Stream destruction

=== Testing SAXPY (Level 1 BLAS) ===
PASS: SAXPY correctness

=== Testing SGEMV (Level 2 BLAS) ===
PASS: SGEMV correctness

=== Testing SGEMM (Level 3 BLAS) ===
PASS: SGEMM correctness

==================================================
ALL TESTS PASSED ✓
==================================================
```

## Test Coverage

| Category     | Operations Tested                                 | Status |
| ------------ | ------------------------------------------------- | ------ |
| Lifecycle    | init, shutdown                                    | ✅      |
| Device Info  | get_device_properties                             | ✅      |
| Memory       | malloc, free, memcpy_h2d, memcpy_d2h, memcpy_d2d  | ✅      |
| Streams      | stream_create, stream_destroy, stream_synchronize | ✅      |
| Level 1 BLAS | saxpy                                             | ✅      |
| Level 2 BLAS | sgemv                                             | ✅      |
| Level 3 BLAS | sgemm                                             | ✅      |

## Adding New Tests

To add tests for additional operations:

1. Add test function to `test_cublas_basic.c`:
   ```c
   int test_new_operation() {
       printf("\n=== Testing NEW_OPERATION ===\n");
       // Test implementation
       return 0;
   }
   ```

2. Call from `main()`:
   ```c
   result |= test_new_operation();
   ```

3. Rebuild and run tests

## Troubleshooting

### "CUDA not found" Error
- Verify CUDA Toolkit is installed: `nvcc --version`
- Check installation path matches build script
- Add CUDA bin directory to PATH

### "Failed to create handle" Error
- Verify NVIDIA GPU is available: `nvidia-smi`
- Update NVIDIA drivers to latest version
- Check GPU compute capability is supported

### Compilation Errors
- Ensure CUDA Toolkit version matches GPU architecture
- Check include paths in build script
- Verify all source files are present

## Performance Testing

For performance benchmarks, see:
- `test_cublas_performance.c` (future)
- `benchmark_all_operations.c` (future)

## Benchmarking & Ranking Policy

The benchmark system ranks **per size/shape bucket**, not globally. This avoids selecting a backend that only wins on tiny sizes but loses at realistic sizes.

- **Size buckets**: 5 size classes × 5 shape classes
- **Warm-up**: run warmup iterations before timing
- **Sampling**: record mean, standard deviation, and p99 latency
- **Stability**: discard noisy samples and retry if variance is high

### Reference Backend Policy

The reference implementation is a **correctness oracle**. It is included in benchmarking runs for validation and can win **only within the bucket where it is fastest** (usually tiny sizes). It is **not** selected as a global default.

If the reference backend wins outside tiny-size buckets, treat it as a signal to:
1. Re-check benchmark configuration (sizes, layout, warmup)
2. Verify backend build flags and vectorization
3. Improve benchmarking detail (more samples, tighter variance control)

This keeps production selection fast while still allowing the reference to win in legitimate micro-size niches.

## Next Steps

After basic tests pass:
1. Test additional Level 1 BLAS operations (daxpy, sscal, sdot, etc.)
2. Test additional Level 2 BLAS operations (dgbmv, ssymv, strmv, etc.)
3. Test additional Level 3 BLAS operations (dgemm, ssymm, strmm, etc.)
4. Test cuSOLVER LAPACK operations (sgetrf, spotrf, sgeqrf, etc.)
5. Test async execution with streams
6. Performance benchmarking vs native cuBLAS
7. Cross-backend validation (cuBLAS vs rocBLAS)
