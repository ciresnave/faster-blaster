# CMake Backend Discovery Modules

This directory contains CMake find modules for automatically locating backend libraries used by faster-blaster.

## Available Modules

### GPU Backends

#### FindOneMKL.cmake ✅
**Purpose**: Intel oneMKL for SYCL/DPC++ GPU acceleration  
**Targets**: Intel Arc/Max GPUs  
**Searches**: Windows, Linux, macOS Intel oneAPI installations  
**Variables**:
- `ONEMKL_FOUND` - System has oneMKL
- `ONEMKL_INCLUDE_DIRS` - Include directories (MKL + SYCL)
- `ONEMKL_LIBRARIES` - Libraries (mkl_sycl, mkl_intel_lp64, mkl_tbb_thread, mkl_core)
- `ONEMKL_VERSION` - Detected version

**Usage**:
```cmake
find_package(OneMKL REQUIRED)
target_include_directories(mytarget PRIVATE ${ONEMKL_INCLUDE_DIRS})
target_link_libraries(mytarget PRIVATE ${ONEMKL_LIBRARIES})
```

#### FindROCBLAS.cmake ✅
**Purpose**: AMD rocBLAS/rocSOLVER for ROCm/HIP GPU acceleration  
**Targets**: AMD Radeon GPUs  
**Searches**: Windows, Linux ROCm installations  
**Variables**:
- `ROCBLAS_FOUND` - System has rocBLAS
- `ROCBLAS_INCLUDE_DIRS` - Include directories (HIP, rocBLAS, rocSOLVER)
- `ROCBLAS_LIBRARIES` - Libraries (rocblas, rocsolver, hipblas)
- `ROCBLAS_VERSION` - Detected version
- `ROCM_ROOT` - ROCm installation root

**Usage**:
```cmake
find_package(ROCBLAS REQUIRED)
target_include_directories(mytarget PRIVATE ${ROCBLAS_INCLUDE_DIRS})
target_link_libraries(mytarget PRIVATE ${ROCBLAS_LIBRARIES})
```

**Note**: For cuBLAS (NVIDIA), use CMake's built-in `find_package(CUDAToolkit)` which is excellent.

### CPU Backends

#### FindMKL.cmake ✅
**Purpose**: Intel MKL for CPU BLAS/LAPACK  
**Targets**: Intel CPUs (optimal), Generic CPUs (good)  
**Searches**: Windows, Linux, macOS Intel oneAPI/MKL installations  
**Variables**:
- `MKL_FOUND` - System has Intel MKL
- `MKL_INCLUDE_DIRS` - Include directory
- `MKL_LIBRARIES` - Libraries (interface + threading + core)
- `MKL_VERSION` - Detected version

**Features**:
- Auto-detects LP64 (64-bit) vs ILP64 interface
- Auto-selects threading layer (intel_thread on Windows, gnu_thread on Linux)
- Provides proper linking line

**Usage**:
```cmake
find_package(MKL REQUIRED)
target_include_directories(mytarget PRIVATE ${MKL_INCLUDE_DIRS})
target_link_libraries(mytarget PRIVATE ${MKL_LIBRARIES})
# May also need OpenMP runtime (iomp5/gomp)
```

#### FindOpenBLAS.cmake ✅
**Purpose**: Portable open-source BLAS/LAPACK  
**Targets**: Any CPU architecture  
**Searches**: Windows (vcpkg, msys64), Linux (package managers), macOS (Homebrew)  
**Variables**:
- `OPENBLAS_FOUND` - System has OpenBLAS
- `OPENBLAS_INCLUDE_DIRS` - Include directory
- `OPENBLAS_LIBRARIES` - Libraries (OpenBLAS, optionally separate LAPACK)
- `OPENBLAS_VERSION` - Detected version

**Features**:
- Handles both bundled LAPACK and separate LAPACK library
- Works with system packages and vcpkg

**Usage**:
```cmake
find_package(OpenBLAS REQUIRED)
target_include_directories(mytarget PRIVATE ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(mytarget PRIVATE ${OPENBLAS_LIBRARIES})
```

#### FindBLIS.cmake ✅
**Purpose**: AMD-optimized BLAS library  
**Targets**: AMD Zen CPUs (optimal), Generic CPUs (good)  
**Searches**: Windows, Linux, macOS BLIS installations  
**Variables**:
- `BLIS_FOUND` - System has BLIS
- `BLIS_INCLUDE_DIRS` - Include directory
- `BLIS_LIBRARIES` - BLIS library
- `BLIS_VERSION` - Detected version

**Note**: BLIS provides BLAS only. LAPACK operations require a separate library (libFLAME or netlib LAPACK).

**Usage**:
```cmake
find_package(BLIS REQUIRED)
target_include_directories(mytarget PRIVATE ${BLIS_INCLUDE_DIRS})
target_link_libraries(mytarget PRIVATE ${BLIS_LIBRARIES})
```

#### FindAOCL.cmake ✅
**Purpose**: AMD Optimizing CPU Libraries (BLIS + libFLAME)  
**Targets**: AMD Zen CPUs (optimal)  
**Searches**: Windows, Linux AOCL installations  
**Variables**:
- `AOCL_FOUND` - System has AOCL
- `AOCL_INCLUDE_DIRS` - Include directories (BLIS + libFLAME)
- `AOCL_LIBRARIES` - Libraries (BLIS + libFLAME)
- `AOCL_VERSION` - Detected version

**Features**:
- Includes both BLIS (BLAS) and libFLAME (LAPACK)
- Full AMD optimization

**Usage**:
```cmake
find_package(AOCL REQUIRED)
target_include_directories(mytarget PRIVATE ${AOCL_INCLUDE_DIRS})
target_link_libraries(mytarget PRIVATE ${AOCL_LIBRARIES})
```

### Generic Framework

#### FindBackendLibraries.cmake 🔧
**Purpose**: Generic framework for backend discovery  
**Provides**: `find_backend_library()` function and platform-specific search paths  

**Usage**:
```cmake
include(FindBackendLibraries)

find_backend_library(
  NAME MyBackend
  HEADERS mybackend.h
  LIBRARIES mybackend mybackend_util
  SEARCH_PATHS "/opt/mybackend" "C:/Program Files/MyBackend"
  ENV_VARS MYBACKEND_ROOT MYBACKEND_HOME
  REQUIRED
)

# Creates:
# - MyBackend_FOUND
# - MyBackend_INCLUDE_DIRS
# - MyBackend_LIBRARIES
```

## Design Principles

### 1. **Zero User Configuration**
All modules search common installation paths automatically. No manual paths needed.

### 2. **Cross-Platform**
Each module knows platform-specific paths:
- Windows: Program Files, vcpkg, msys64
- Linux: /opt, /usr/local, package managers
- macOS: Homebrew, MacPorts

### 3. **Version Detection**
Modules attempt to extract version information from headers or installation paths.

### 4. **Graceful Degradation**
Optional backends don't break the build. Use `REQUIRED` keyword only for mandatory backends.

### 5. **Informative Output**
Each module prints what it found with helpful messages.

### 6. **Standard CMake Patterns**
All modules use `FindPackageHandleStandardArgs` for consistent behavior.

## Integration Example

```cmake
# In main CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# GPU backends (optional)
find_package(OneMKL QUIET)
if(ONEMKL_FOUND)
    target_link_libraries(mylib PRIVATE ${ONEMKL_LIBRARIES})
endif()

find_package(ROCBLAS QUIET)
if(ROCBLAS_FOUND)
    target_link_libraries(mylib PRIVATE ${ROCBLAS_LIBRARIES})
endif()

# CPU backends (fallback chain)
find_package(MKL QUIET)
if(NOT MKL_FOUND)
    find_package(OpenBLAS QUIET)
endif()

if(MKL_FOUND)
    target_link_libraries(mylib PRIVATE ${MKL_LIBRARIES})
elseif(OPENBLAS_FOUND)
    target_link_libraries(mylib PRIVATE ${OPENBLAS_LIBRARIES})
else()
    message(WARNING "No optimized BLAS found, using reference implementation")
endif()
```

## Environment Variables

Users can override search paths by setting environment variables:

**GPU**:
- `MKLROOT` or `ONEAPI_ROOT` - Intel oneMKL
- `ROCM_PATH` or `HIP_PATH` - AMD ROCm
- `CUDA_PATH` or `CUDA_HOME` - NVIDIA CUDA (built-in CMake)

**CPU**:
- `MKLROOT` - Intel MKL
- `OPENBLAS_ROOT` - OpenBLAS
- `BLIS_ROOT` - BLIS
- `AOCL_ROOT` - AMD AOCL

Example:
```bash
# Linux
export MKLROOT=/opt/intel/oneapi/mkl/latest
cmake ..

# Windows PowerShell
$env:MKLROOT = "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
cmake ..
```

## Adding New Backends

To add a new backend:

1. Create `cmake/FindYourBackend.cmake`
2. Follow the pattern from existing modules
3. Define search paths for all platforms
4. Find headers and libraries
5. Extract version if possible
6. Use `FindPackageHandleStandardArgs`
7. Print helpful status messages

Template:
```cmake
# FindYourBackend.cmake
set(YOURBACKEND_SEARCH_PATHS ...)
find_path(YOURBACKEND_INCLUDE_DIR ...)
find_library(YOURBACKEND_LIBRARY ...)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(YourBackend ...)
```

## Testing Find Modules

```bash
# Clear cache and reconfigure
rm CMakeCache.txt
cmake ..

# Check output for "Found XYZ:" messages
# Verify include paths and library paths are correct
```

## Maintenance

**When updating**:
- Add new version paths as they're released
- Test on all supported platforms
- Update this README with changes

**Current Status** (Dec 2025):
- ✅ OneMKL - Tested on Windows with oneAPI 2025.3
- ✅ ROCBLAS - Created, not yet tested
- ✅ MKL - Created, not yet tested
- ✅ OpenBLAS - Created, not yet tested
- ✅ BLIS - Created, not yet tested
- ✅ AOCL - Created, not yet tested

---

**Project**: faster-blaster  
**Module Version**: 1.0  
**Last Updated**: December 14, 2025
