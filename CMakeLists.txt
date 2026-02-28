cmake_minimum_required(VERSION 3.15)

# Auto-configure vcpkg toolchain if not already set
# This allows `cmake -B build` to work without --preset or -DCMAKE_TOOLCHAIN_FILE
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "vcpkg toolchain file")
        message(STATUS "Auto-detected vcpkg: $ENV{VCPKG_ROOT}")
    elseif(EXISTS "C:/libraries/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/libraries/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "vcpkg toolchain file")
        message(STATUS "Auto-detected vcpkg: C:/libraries/vcpkg")
    endif()
endif()

project(faster-blaster VERSION 0.1.0 LANGUAGES C CXX)

# Build backends from source with hardware-specific optimizations
# Set FB_BUILD_BACKENDS_FROM_SOURCE=OFF to use system libraries instead
include(cmake/BuildBackendsFromSource.cmake)

# Include backend detection and installer (for system-installed backends)
include(cmake/BackendInstaller.cmake)

# C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_STATISTICS "Enable runtime statistics collection" ON)

# GPU Backend options (AUTO = enable if dependencies found)
set(FB_ENABLE_CUDA "AUTO" CACHE STRING "Enable NVIDIA cuBLAS (ON/OFF/AUTO)")
set(FB_ENABLE_ROCM "AUTO" CACHE STRING "Enable AMD rocBLAS (ON/OFF/AUTO)")
set(FB_ENABLE_ONEMKL "OFF" CACHE STRING "Enable Intel oneMKL GPU (ON/OFF/AUTO)") # Disabled temporarily - member order errors
set(FB_ENABLE_METAL "AUTO" CACHE STRING "Enable Apple Metal (ON/OFF/AUTO)")

# Legacy CPU backend options (kept for compatibility)
option(ENABLE_MKL "Enable Intel MKL backend" ON)
option(ENABLE_CUBLAS "Enable NVIDIA cuBLAS backend (deprecated: use FB_ENABLE_CUDA)" ON)
option(ENABLE_ROCBLAS "Enable AMD rocBLAS backend (deprecated: use FB_ENABLE_ROCM)" ON)
option(ENABLE_HIPBLAS "Enable AMD hipBLAS backend" ON)
option(ENABLE_ONEMKL "Enable Intel oneMKL GPU backend" OFF)
option(ENABLE_ACCELERATE "Enable Apple Accelerate backend" ON)
option(ENABLE_BLIS "Enable BLIS backend" ON)
option(ENABLE_OPENBLAS "Enable OpenBLAS backend" ON)
option(ENABLE_AOCL "Enable AMD AOCL backend" ON)

# C23 standard required
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform-specific settings
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add our custom Find modules to the search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find cpuinfo library
find_package(cpuinfo QUIET)
if(NOT cpuinfo_FOUND)
    message(STATUS "cpuinfo not found, will fetch from source")
    include(FetchContent)
    # Disable cpuinfo tools and tests to avoid build issues
    set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
    set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        cpuinfo
        GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(cpuinfo)
endif()

# Find OpenCL
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_INCLUDE_DIRS}")
    set(FB_ENABLE_OPENCL ON)
    add_definitions(-DFB_ENABLE_OPENCL)
else()
    message(FATAL_ERROR "OpenCL not found. Please install OpenCL SDK.")
endif()

# Find CLBlast (OpenCL BLAS library)
# vcpkg uses CLBlast (capital C and B) for the package name
find_package(CLBlast CONFIG QUIET)
if(NOT CLBlast_FOUND)
    # Try lowercase as fallback
    find_package(clblast CONFIG QUIET)
endif()

if(CLBlast_FOUND OR clblast_FOUND)
    message(STATUS "✓ Found CLBlast")
    set(CLBLAST_AVAILABLE ON)
else()
    message(STATUS "⚠ CLBlast not found (install via: vcpkg install clblast)")
    message(STATUS "  CLBlast support will be disabled")
    set(CLBLAST_AVAILABLE OFF)
endif()

# ============================================================================
# GPU Backend Auto-Detection
# ============================================================================

# CUDA Detection (for cuBLAS plugin)
set(CUDA_AVAILABLE OFF)
if(FB_ENABLE_CUDA STREQUAL "AUTO" OR FB_ENABLE_CUDA OR ENABLE_CUBLAS)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CUDA_AVAILABLE ON)
        
        # Check if nvcc compiler is available
        find_program(CUDA_NVCC_EXECUTABLE nvcc HINTS ${CUDAToolkit_BIN_DIR})
        if(CUDA_NVCC_EXECUTABLE)
            # Allow CUDA language with clang-cl (MSVC-mode Clang); disable only for true MSVC (cl.exe)
            if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND MSVC)
                # clang-cl (MSVC-mode Clang): cuBLAS libraries work fine — just don't enable
                # the CUDA language since nvcc does not support Clang-cl as a host compiler on
                # Windows.  The cuBLAS plugin .c files are compiled by clang-cl, not nvcc.
                message(STATUS "✓ CUDA compiler (nvcc) found: ${CUDA_NVCC_EXECUTABLE}")
                message(STATUS "  Note: CUDA language not enabled with clang-cl (nvcc unsupported host on Windows)")
                message(STATUS "  cuBLAS plugin links against CUDAToolkit::cublas directly")
                set(CMAKE_CUDA_STANDARD 20)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
                message(STATUS "✓ CUDA compiler (nvcc) found and enabled with clang-cl host: ${CUDA_NVCC_EXECUTABLE}")
            elseif(NOT MSVC)
                enable_language(CUDA)
                set(CMAKE_CUDA_STANDARD 20)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
                message(STATUS "✓ CUDA compiler (nvcc) found and enabled: ${CUDA_NVCC_EXECUTABLE}")
            else()
                message(STATUS "✓ CUDA compiler (nvcc) found: ${CUDA_NVCC_EXECUTABLE}")
                message(STATUS "  Note: CUDA language not enabled with MSVC cl.exe generator (cuBLAS tests disabled)")
            endif()
        else()
            message(STATUS "⚠ CUDA libraries found but nvcc compiler not available")
        endif()
        
        message(STATUS "✓ CUDA Toolkit ${CUDAToolkit_VERSION} detected")
        message(STATUS "  cuBLAS plugin will be enabled")
    else()
        if(FB_ENABLE_CUDA STREQUAL "ON")
            message(FATAL_ERROR "✗ CUDA Toolkit not found but FB_ENABLE_CUDA=ON")
        else()
            message(STATUS "✗ CUDA Toolkit not found - cuBLAS plugin disabled")
        endif()
    endif()
endif()

# ROCm Detection (for rocBLAS plugin)
set(ROCM_AVAILABLE OFF)
if(FB_ENABLE_ROCM STREQUAL "AUTO" OR FB_ENABLE_ROCM)
    # Use our custom FindROCBLAS module
    find_package(ROCBLAS QUIET)
    
    if(ROCBLAS_FOUND)
        set(ROCM_AVAILABLE ON)
        message(STATUS "✓ ROCm ${ROCBLAS_VERSION} detected at ${ROCM_ROOT}")
        message(STATUS "  rocBLAS plugin will be enabled")
        
        # Enable HIP compilation for hip_detect.c
        # HIP requires AMD's clang compiler, not MSVC's cl.exe
        if(WIN32)
            # On Windows, try to find hipcc in standard ROCm installation path
            if(NOT HIPCC_EXECUTABLE)
                # Use ROCM_ROOT from FindROCBLAS, or search standard paths
                set(HIPCC_SEARCH_PATHS)
                if(ROCM_ROOT)
                    list(APPEND HIPCC_SEARCH_PATHS "${ROCM_ROOT}/bin")
                endif()
                list(APPEND HIPCC_SEARCH_PATHS 
                    "C:/Program Files/AMD/ROCm/*/bin"
                    "$ENV{ROCM_PATH}/bin"
                )
                
                find_program(HIPCC_EXECUTABLE
                    NAMES hipcc hipcc.bat hipcc.exe
                    PATHS ${HIPCC_SEARCH_PATHS}
                    NO_DEFAULT_PATH
                )
            endif()
            
            if(HIPCC_EXECUTABLE)
                # Visual Studio generator doesn't support HIP language
                # We'll compile hip_detect.c with a custom command instead
                set(HIP_COMPILER_AVAILABLE ON)
                message(STATUS "  Found hipcc: ${HIPCC_EXECUTABLE}")
                message(STATUS "  HIP source files will be compiled using custom commands")
            else()
                set(HIP_COMPILER_AVAILABLE OFF)
                message(STATUS "  hipcc not found - HIP detection will be disabled")
                message(STATUS "  Tip: Install ROCm SDK to enable AMD GPU detection")
            endif()
        else()
            # On Linux/Unix, try to use HIP language support
            include(CheckLanguage)
            check_language(HIP)
            if(CMAKE_HIP_COMPILER)
                # Only enable if generator supports it (not Visual Studio)
                if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
                    enable_language(HIP)
                    set(HIP_LANGUAGE_AVAILABLE ON)
                    set(HIP_COMPILER_AVAILABLE ON)
                else()
                    # Fallback to custom command on Linux with VS generator too
                    set(HIPCC_EXECUTABLE ${CMAKE_HIP_COMPILER})
                    set(HIP_COMPILER_AVAILABLE ON)
                    set(HIP_LANGUAGE_AVAILABLE OFF)
                endif()
            else()
                set(HIP_COMPILER_AVAILABLE OFF)
                set(HIP_LANGUAGE_AVAILABLE OFF)
            endif()
        endif()
    else()
        if(FB_ENABLE_ROCM STREQUAL "ON")
            message(FATAL_ERROR "✗ ROCm not found but FB_ENABLE_ROCM=ON")
        else()
            message(STATUS "✗ ROCm not found - rocBLAS plugin disabled")
        endif()
    endif()
endif()

# Intel oneAPI Detection (for oneMKL GPU plugin)
set(ONEMKL_AVAILABLE OFF)
if(FB_ENABLE_ONEMKL STREQUAL "AUTO" OR FB_ENABLE_ONEMKL)
    # Use our custom FindOneMKL module
    find_package(OneMKL QUIET)
    
    if(ONEMKL_FOUND)
        set(ONEMKL_AVAILABLE ON)
        message(STATUS "✓ Intel oneMKL ${ONEMKL_VERSION} detected")
        message(STATUS "  oneMKL GPU plugin will be enabled")
    else()
        if(FB_ENABLE_ONEMKL STREQUAL "ON")
            message(FATAL_ERROR "✗ Intel oneMKL not found but FB_ENABLE_ONEMKL=ON")
        else()
            message(STATUS "✗ Intel oneMKL not found - oneMKL GPU plugin disabled")
        endif()
    endif()
endif()

# Metal Detection (for Metal GPU plugin - macOS only)
set(METAL_AVAILABLE OFF)
if(APPLE)
    if(FB_ENABLE_METAL STREQUAL "AUTO" OR FB_ENABLE_METAL)
        # Metal is always available on macOS 10.11+
        set(METAL_AVAILABLE ON)
        message(STATUS "✓ Metal framework available (macOS)")
        message(STATUS "  Metal GPU plugin will be enabled")
    endif()
else()
    if(FB_ENABLE_METAL STREQUAL "ON")
        message(FATAL_ERROR "✗ Metal only available on macOS but FB_ENABLE_METAL=ON")
    endif()
endif()

# Summary
message(STATUS "")
message(STATUS "GPU Backend Configuration:")
message(STATUS "  NVIDIA cuBLAS:  ${CUDA_AVAILABLE}")
message(STATUS "  AMD rocBLAS:    ${ROCM_AVAILABLE}")
message(STATUS "  Intel oneMKL:   ${ONEMKL_AVAILABLE}")
message(STATUS "  Apple Metal:    ${METAL_AVAILABLE}")
message(STATUS "")

# ============================================================================
# CPU Backend Auto-Detection
# ============================================================================

# Intel MKL Detection (for MKL CPU plugin)
set(MKL_CPU_AVAILABLE OFF)
if(ENABLE_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND)
        set(MKL_CPU_AVAILABLE ON)
        message(STATUS "✓ Intel MKL ${MKL_VERSION} detected")
        message(STATUS "  MKL CPU plugin will be enabled")
    else()
        message(STATUS "✗ Intel MKL not found - MKL CPU plugin disabled")
        set(ENABLE_MKL OFF)
    endif()
endif()

# OpenBLAS Detection (for OpenBLAS plugin)
set(OPENBLAS_CPU_AVAILABLE OFF)
if(ENABLE_OPENBLAS)
    find_package(OpenBLAS QUIET)
    if(OPENBLAS_FOUND)
        set(OPENBLAS_CPU_AVAILABLE ON)
        message(STATUS "✓ OpenBLAS ${OPENBLAS_VERSION} detected")
        message(STATUS "  OpenBLAS plugin will be enabled")
    else()
        message(STATUS "✗ OpenBLAS not found - OpenBLAS plugin disabled")
        set(ENABLE_OPENBLAS OFF)
    endif()
endif()

# BLIS Detection (for BLIS plugin)
set(BLIS_CPU_AVAILABLE OFF)
if(ENABLE_BLIS)
    find_package(BLIS QUIET)
    if(BLIS_FOUND)
        set(BLIS_CPU_AVAILABLE ON)
        message(STATUS "✓ BLIS ${BLIS_VERSION} detected")
        message(STATUS "  BLIS plugin will be enabled")
    else()
        message(STATUS "✗ BLIS not found - BLIS plugin disabled")
        set(ENABLE_BLIS OFF)
    endif()
endif()

# AMD AOCL Detection (for AOCL plugin - includes BLIS + libFLAME)
set(AOCL_CPU_AVAILABLE OFF)
if(ENABLE_AOCL)
    find_package(AOCL QUIET)
    if(AOCL_FOUND)
        set(AOCL_CPU_AVAILABLE ON)
        message(STATUS "✓ AMD AOCL ${AOCL_VERSION} detected")
        message(STATUS "  AOCL plugin will be enabled")
    else()
        message(STATUS "✗ AMD AOCL not found - AOCL plugin disabled")
        set(ENABLE_AOCL OFF)
    endif()
endif()

# Apple Accelerate (always available on macOS)
set(ACCELERATE_CPU_AVAILABLE OFF)
if(APPLE AND ENABLE_ACCELERATE)
    set(ACCELERATE_CPU_AVAILABLE ON)
    message(STATUS "✓ Apple Accelerate framework available")
    message(STATUS "  Accelerate plugin will be enabled")
endif()

message(STATUS "")
message(STATUS "CPU Backend Configuration:")
message(STATUS "  Intel MKL:      ${MKL_CPU_AVAILABLE}")
message(STATUS "  OpenBLAS:       ${OPENBLAS_CPU_AVAILABLE}")
message(STATUS "  BLIS:           ${BLIS_CPU_AVAILABLE}")
message(STATUS "  AMD AOCL:       ${AOCL_CPU_AVAILABLE}")
message(STATUS "  Accelerate:     ${ACCELERATE_CPU_AVAILABLE}")
message(STATUS "")

# ============================================================================
# End CPU Backend Auto-Detection
# ============================================================================

# Legacy compatibility
if(ENABLE_CUBLAS AND NOT CUDA_AVAILABLE)
    set(ENABLE_CUBLAS OFF)
endif()
if(ENABLE_ROCBLAS AND NOT ROCM_AVAILABLE)
    set(ENABLE_ROCBLAS OFF)
endif()

# ============================================================================
# End GPU Backend Auto-Detection
# ============================================================================

# Core library sources
set(CORE_SOURCES
    src/core/dispatch.c
    src/core/plugin_registry.c
    src/core/plugin_init.c
    src/core/backend_instance.c
    src/core/device_api.c
    src/core/dispatch_unified.c
    # Real device detection implementations (using cpu_detect.c and gpu_detect.c)
    src/core/device_registry.c
    src/core/compute_manager.c
    # Device memory manager for GPU operation chaining
    src/core/device_memory_manager.c
    # Vtable auto-fill system (Phases 1-5)
    src/core/vtable_autofill.c
    src/core/vtable_wrappers_gemm.c
    src/core/vtable_wrappers_normalization.c
    src/core/vtable_wrappers_reduction.c
    src/core/vtable_dispatchers_gemm.c
    src/core/vtable_wrappers_batched.c
    src/core/vtable_wrappers_strided_to_array.c
    src/core/vtable_wrappers_precision_promotion.c
    src/core/ranked_dispatch.c
    # Old stubs (replaced with real implementations above)
    # src/core/device_registry_stub.c
    # src/core/compute_manager_stub.c
    # TODO: Fix calibration implementation after redesign
    # src/core/calibration.c
    # TODO: Fix cpuinfo linking on Windows
    # src/core/hardware_detect.c
    src/core/timing.c
    # TODO: Re-enable after removing fb_dtype_t dependency
    # src/core/test_data.c
)

# Benchmark system sources (Phase 3.0)
set(BENCHMARK_SOURCES
    src/benchmark_classify.c
    src/benchmark_fingerprint.c
    src/benchmark_cache.c
    src/benchmark_execute.c
    src/dispatch_tables.c
    src/auto_benchmark.c
)

# API wrapper sources
set(API_SOURCES
    src/api/level1.c
    src/api/level2.c
    src/api/level3.c
    # TODO: Old init.c conflicts with dispatch_unified.c's fb_init - using new implementation
    # src/api/init.c
)

# Device detection sources
set(DEVICE_DETECTION_SOURCES
    src/device_detection/cpu_detect.c
    src/device_detection/gpu_detect.c
    src/device_detection/opencl_detect.c
)

# Add platform-specific GPU detection (separate files to avoid header conflicts)
if(CUDA_AVAILABLE)
    list(APPEND DEVICE_DETECTION_SOURCES src/device_detection/cuda_detect.c)
endif()

# Add HIP detection if HIP compiler is available
# Note: HIP files must be compiled with hipcc/AMD clang, not MSVC cl.exe
if(ROCM_AVAILABLE AND HIP_COMPILER_AVAILABLE)
    if(HIP_LANGUAGE_AVAILABLE)
        # Use CMake's native HIP language support (Ninja, Makefiles)
        list(APPEND DEVICE_DETECTION_SOURCES src/device_detection/hip_detect.c)
        set_source_files_properties(src/device_detection/hip_detect.c 
            PROPERTIES LANGUAGE HIP)
        message(STATUS "  HIP detection will use native HIP language support")
    else()
        # Compile hip_detect.c with custom command (Visual Studio generator)
        # hipcc.bat doesn't properly handle paths with spaces, so we'll use clang directly
        set(HIP_DETECT_OBJ "${CMAKE_CURRENT_BINARY_DIR}/hip_detect${CMAKE_C_OUTPUT_EXTENSION}")
        
        # Find clang.exe in ROCm installation
        find_program(HIP_CLANG_EXECUTABLE
            NAMES clang.exe clang
            PATHS "${ROCM_ROOT}/bin" "${ROCM_ROOT}/llvm/bin"
            NO_DEFAULT_PATH
        )
        
        if(NOT HIP_CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find clang.exe in ROCm installation")
        endif()
        
        # Get HIP includes and defines from hipcc environment
        set(HIP_CLANG_INCLUDES
            "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-I${ROCM_ROOT}/include"
            "-I${ROCM_ROOT}/include/hip"
        )
        
        add_custom_command(
            OUTPUT ${HIP_DETECT_OBJ}
            COMMAND "${HIP_CLANG_EXECUTABLE}"
                -c
                -o "${HIP_DETECT_OBJ}"
                -x hip
                -O3
                -DFB_ENABLE_HIP
                -D__HIP_PLATFORM_AMD__=1
                -D__HIP_ROCclr__=1
                ${HIP_CLANG_INCLUDES}
                "${CMAKE_CURRENT_SOURCE_DIR}/src/device_detection/hip_detect.c"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/device_detection/hip_detect.c
            COMMENT "Compiling hip_detect.c with HIP clang"
            VERBATIM
        )
        
        # Add object file to sources
        list(APPEND DEVICE_DETECTION_SOURCES ${HIP_DETECT_OBJ})
        message(STATUS "  HIP detection will compile using clang directly: ${HIP_CLANG_EXECUTABLE}")
    endif()
elseif(ROCM_AVAILABLE)
    message(STATUS "  HIP detection disabled: hipcc not found")
endif()

# Backend sources
set(BACKEND_SOURCES
    src/backends/reference.c
    src/backends/openblas_backend.c
    src/backends/backend_loader.c
    src/backends/backend_matcher.c
    # CPU trait implementations
    src/backends/cpu/blis_trait_impl.c
    src/backends/cpu/openblas_trait_impl.c
    src/backends/cpu/reference_trait_impl.c
)

# Plugin sources (new architecture)
set(PLUGIN_SOURCES
    # CPU plugins
    src/plugins/plugin_aocl_blis.c
    src/plugins/plugin_standard_blis.c
    src/plugins/plugin_openblas.c
    src/plugins/plugin_mkl.c
    src/plugins/plugin_accelerate.c
    # GPU plugins
    # cuBLAS plugins enabled
    src/backends/gpu/cublas_trait_impl.c  # Real trait impl (replaces backend_cublas.c stub)
    src/plugins/plugin_cublas.c
    src/plugins/cublas_vtable_wrappers.c
    src/plugins/cublas_smart_wrappers.c
    src/plugins/cublas_complete_smart_wrappers.c
    src/plugins/cublas_blas_level1_smart.c
    src/plugins/cublas_blas_level2_smart.c
    src/plugins/cublas_blas_level3_smart.c
    src/plugins/plugin_rocblas.c
    # src/plugins/rocblas_vtable_wrappers.c  # Requires HIP compiler - needs add_custom_command like hip_detect.c
    src/plugins/plugin_onemkl.c
    src/plugins/plugin_metal.c
    src/plugins/plugin_clblast.c
    src/plugins/clblast_vtable_wrappers.c
    src/plugins/plugin_clblas.c
    # Reference backend: always-available correctness oracle and last-resort fallback
    src/plugins/plugin_reference.c
)

# Judge module sources (precision/accuracy profiling for operation routing)
set(JUDGE_SOURCES
    src/judge/judge.c
    src/judge/judge_norm.c
    src/judge/judge_metadata.c
    src/judge/judge_metadata_table.c
    src/judge/judge_corpus.c
    src/judge/judge_direct.c
    src/judge/judge_index.c
    src/judge/judge_factorization.c
    src/judge/judge_solve.c
    src/judge/judge_spectral.c
    src/judge/judge_profile.c
    src/judge/judge_store.c
    src/judge/judge_select.c
)

# Add backend source files based on what's enabled
# (Note: target_*() commands are called after add_library() below)
if(MKL_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/mkl_backend.c)
    message(STATUS "Adding Intel MKL CPU backend source")
endif()

if(BLIS_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/blis_backend.c)
    message(STATUS "Adding BLIS CPU backend source")
endif()

if(OPENBLAS_CPU_AVAILABLE)
    # OpenBLAS backend already has sources in BACKEND_SOURCES
    message(STATUS "Adding OpenBLAS CPU backend source")
endif()

if(AOCL_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/aocl_backend.c)
    message(STATUS "Adding AMD AOCL CPU backend source")
endif()

if(ACCELERATE_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/accelerate_backend.c)
    message(STATUS "Adding Apple Accelerate backend source")
endif()

if(CUDA_AVAILABLE)
    # Add GPU backend trait implementations
    file(GLOB CUBLAS_SOURCES src/backends/gpu/*cublas*.c)
    # Exclude trait_impl files - they use static linking and are only for tests
    # Exclude vtable_wrappers and smart wrapper files - they require fb_cublas_trait which would create static CUDA imports
    # Use only cublas_backend.c which uses pure dynamic loading for CUDA functions
    list(FILTER CUBLAS_SOURCES EXCLUDE REGEX "trait_impl\\.c$|vtable_wrappers\\.c$|smart.*\\.c$")
    list(APPEND BACKEND_SOURCES ${CUBLAS_SOURCES})
endif()

if(ROCM_AVAILABLE AND HIP_COMPILER_AVAILABLE)
    # rocBLAS backend compilation attempted but rocSOLVER API differs significantly from cuSOLVER
    # Completed conversions:
    #   - CUDA → HIP runtime types (cudaMalloc → hipMalloc, etc.)
    #   - cuBLAS → rocBLAS API  (cublasSgemm → rocblas_sgemm, etc.)
    #   - Complex types (cuDoubleComplex → rocblas_double_complex)
    #   - Constants (CUBLAS_OP_N → rocblas_operation_none, etc.)
    #   - HIP compilation setup (add_custom_command for clang)
    #
    # rocBLAS backend enabled with BLAS support
    # LAPACK support temporarily disabled pending complete rocSOLVER API conversion
    # (rocSOLVER API differs from cuSOLVER - no _bufferSize functions, simpler workspace management)
    
    message(STATUS "rocBLAS backend enabled (BLAS Level 1-3 + LAPACK via rocSOLVER)")
    message(STATUS "   Note: rocSOLVER provides LU, Cholesky, QR, SVD, and eigenvalue solvers")
endif()

if(ONEMKL_AVAILABLE)
    # Add oneMKL GPU backend trait implementations
    file(GLOB ONEMKL_SOURCES src/backends/gpu/*onemkl*.cpp)
    list(APPEND BACKEND_SOURCES ${ONEMKL_SOURCES})
endif()

if(FB_ENABLE_OPENCL)
    # Add CLBlast GPU backend trait implementation
    file(GLOB CLBLAST_SOURCES src/backends/gpu/*clblast*.c)
    list(APPEND BACKEND_SOURCES ${CLBLAST_SOURCES})
    message(STATUS "Adding CLBlast backend trait implementation")
endif()

if(ENABLE_AOCL)
    list(APPEND BACKEND_SOURCES src/backends/aocl_backend.c)
endif()

# Create main library
add_library(faster-blaster
    ${CORE_SOURCES}
    ${BENCHMARK_SOURCES}
    ${API_SOURCES}
    ${DEVICE_DETECTION_SOURCES}
    ${BACKEND_SOURCES}
    ${PLUGIN_SOURCES}
    ${JUDGE_SOURCES}
)

# Enable automatic DLL export on Windows
set_target_properties(faster-blaster PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(faster-blaster
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
)

target_link_libraries(faster-blaster
    PRIVATE
        cpuinfo_internals
        OpenCL::OpenCL
)

# Link GPU backend libraries if available
if(CUDA_AVAILABLE)
    target_link_libraries(faster-blaster PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::nvml  # For NVML monitoring in cuda_detect.c
    )
    target_include_directories(faster-blaster PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_CUDA)
    
    # Delay-load CUDA DLLs on Windows to prevent load-time dependency failures
    if(MSVC)
        target_link_options(faster-blaster PRIVATE
            /DELAYLOAD:cudart64_13.dll
            /DELAYLOAD:cudart64_12.dll
            /DELAYLOAD:cudart64_11.dll
            /DELAYLOAD:cublas64_13.dll
            /DELAYLOAD:cublas64_12.dll
            /DELAYLOAD:cublas64_11.dll
            /DELAYLOAD:cusolver64_13.dll
            /DELAYLOAD:cusolver64_12.dll
            /DELAYLOAD:cusolver64_11.dll
            /DELAYLOAD:nvml.dll
        )
        # Link delayimp.lib required for delay loading
        target_link_libraries(faster-blaster PRIVATE delayimp.lib)
    endif()
    
    # Ensure cuda_detect.c gets CUDA includes
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/device_detection/cuda_detect.c")
        set_source_files_properties(src/device_detection/cuda_detect.c PROPERTIES
            COMPILE_DEFINITIONS "FB_ENABLE_CUDA"
        )
    endif()
    
    message(STATUS "Linking CUDA libraries for cuBLAS plugin with GPU detection")
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

if(ROCM_AVAILABLE)
    # Link ROCm libraries
    target_link_libraries(faster-blaster PRIVATE ${ROCBLAS_LIBRARIES})
    target_include_directories(faster-blaster PRIVATE ${ROCBLAS_INCLUDE_DIRS})
    
    # Delay-load ROCm/HIP DLLs on Windows
    if(MSVC)
        target_link_options(faster-blaster PRIVATE
            /DELAYLOAD:amdhip64_6.dll
            /DELAYLOAD:amdhip64_5.dll
            /DELAYLOAD:rocblas.dll
        )
    endif()
    
    if(HIP_COMPILER_AVAILABLE)
        # HIP compiler available - enable full HIP support
        if(HIP_INCLUDE_DIR)
            target_include_directories(faster-blaster PRIVATE ${HIP_INCLUDE_DIR})
            
            # Link HIP runtime library for hip_detect.c
            find_library(HIP_LIBRARY
                NAMES amdhip64
                PATHS "${ROCM_ROOT}/lib"
                NO_DEFAULT_PATH
            )
            if(HIP_LIBRARY)
                target_link_libraries(faster-blaster PRIVATE ${HIP_LIBRARY})
                message(STATUS "  Linked HIP runtime: ${HIP_LIBRARY}")
            else()
                message(WARNING "HIP runtime library (amdhip64.lib) not found")
            endif()
            
            # If using native HIP language, set compile definitions
            if(HIP_LANGUAGE_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/device_detection/hip_detect.c")
                set_source_files_properties(src/device_detection/hip_detect.c PROPERTIES
                    COMPILE_DEFINITIONS "FB_ENABLE_HIP;__HIP_PLATFORM_AMD__"
                )
            endif()
            # Note: Custom command already includes defines, no need to set them again
            
            # backend_loader.c should NOT have FB_ENABLE_HIP when CUDA is also enabled (header conflicts)
            if(NOT CUDA_AVAILABLE)
                set_source_files_properties(src/backends/backend_loader.c PROPERTIES
                    COMPILE_DEFINITIONS "FB_ENABLE_HIP;__HIP_PLATFORM_AMD__"
                )
            endif()
        endif()
        
        # Enable HIP globally for C files that need HIP API
        target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_HIP)
        
        message(STATUS "ROCm HIP support enabled with HIP compiler")
    else()
        message(STATUS "ROCm libraries linked but HIP detection disabled (no HIP compiler)")
    endif()
endif()

if(ONEMKL_AVAILABLE)
    target_include_directories(faster-blaster PRIVATE ${ONEMKL_INCLUDE_DIRS})
    target_link_libraries(faster-blaster PRIVATE ${ONEMKL_LIBRARIES})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_ONEMKL)
    # Set C++20 for oneMKL files (required for designated initializers)
    set_source_files_properties(${ONEMKL_SOURCES} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    message(STATUS "Intel oneAPI support enabled for oneMKL GPU plugin")
endif()

if(METAL_AVAILABLE)
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_METAL)
    message(STATUS "Metal framework enabled for Metal GPU plugin")
endif()

# Link CPU backend libraries if available
if(MKL_CPU_AVAILABLE)
    target_include_directories(faster-blaster PRIVATE ${MKL_INCLUDE_DIRS})
    target_link_libraries(faster-blaster PRIVATE ${MKL_LIBRARIES})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_MKL)
    message(STATUS "Linking Intel MKL CPU libraries")
endif()

if(OPENBLAS_CPU_AVAILABLE)
    target_include_directories(faster-blaster PRIVATE ${OPENBLAS_INCLUDE_DIRS})
    target_link_libraries(faster-blaster PRIVATE ${OPENBLAS_LIBRARIES})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_OPENBLAS)
    message(STATUS "Linking OpenBLAS libraries")
endif()

if(BLIS_CPU_AVAILABLE)
    target_include_directories(faster-blaster PRIVATE ${BLIS_INCLUDE_DIRS})
    target_link_libraries(faster-blaster PRIVATE ${BLIS_LIBRARIES})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_BLIS)
    message(STATUS "Linking BLIS libraries")
endif()

if(AOCL_CPU_AVAILABLE)
    target_include_directories(faster-blaster PRIVATE ${AOCL_INCLUDE_DIRS})
    target_link_libraries(faster-blaster PRIVATE ${AOCL_LIBRARIES})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_AOCL)
    message(STATUS "Linking AMD AOCL libraries")
endif()

if(ACCELERATE_CPU_AVAILABLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(faster-blaster PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_ACCELERATE)
    message(STATUS "Linking Apple Accelerate framework")
endif()

# Platform-specific libraries
if(WIN32)
    target_link_libraries(faster-blaster PRIVATE)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(faster-blaster PRIVATE pthread dl m)
elseif(APPLE)
    target_link_libraries(faster-blaster PRIVATE pthread dl m)
endif()

# OpenCL libraries
if(OpenCL_FOUND)
    target_link_libraries(faster-blaster PRIVATE OpenCL::OpenCL)
    message(STATUS "Linking OpenCL: ${OpenCL_LIBRARIES}")
endif()

# CLBlast library
if(CLBLAST_AVAILABLE)
    target_link_libraries(faster-blaster PRIVATE clblast)
    message(STATUS "Linking CLBlast")
endif()

# Compiler warnings
# Note: MSVC variable is true even for plain clang.exe on Windows (SIMULATE_ID=MSVC).
# Use CMAKE_C_COMPILER_FRONTEND_VARIANT to distinguish GNU-style clang from cl.exe/clang-cl.
if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # Genuine cl.exe or clang-cl — MSVC flag syntax
    target_compile_options(faster-blaster PRIVATE /W4)
else()
    # Regular clang or GCC — GNU flag syntax
    target_compile_options(faster-blaster PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Statistics compilation flag
if(ENABLE_STATISTICS)
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_STATISTICS)
endif()

# Backends are now added directly to BACKEND_SOURCES above
# No subdirectories needed - all backends are compiled into main library

# Tests
if(BUILD_TESTS)
    enable_testing()
    # Auto-copy the DLL into the tests output directory so tests always load the fresh build
    add_custom_command(TARGET faster-blaster POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            ${CMAKE_CURRENT_BINARY_DIR}/tests/$<TARGET_FILE_NAME:faster-blaster>
        COMMENT "Copying faster-blaster DLL to tests directory"
    )
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    # add_subdirectory(benchmarks)  # TODO: Create benchmarks
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS faster-blaster
    EXPORT faster-blaster-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT faster-blaster-targets
    FILE faster-blaster-targets.cmake
    NAMESPACE FasterBlaster::
    DESTINATION lib/cmake/faster-blaster
)

# CMake package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/faster-blaster-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/faster-blaster-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/faster-blaster-config.cmake"
    INSTALL_DESTINATION lib/cmake/faster-blaster
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/faster-blaster-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/faster-blaster-config-version.cmake"
    DESTINATION lib/cmake/faster-blaster
)

# Package generation
set(CPACK_PACKAGE_NAME "faster-blaster")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Multi-Backend BLAS Abstraction")
set(CPACK_PACKAGE_CONTACT "maintainer@example.com")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Faster-BLASTER Configuration Summary ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Shared libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "")
message(STATUS "Optional Features:")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Statistics: ${ENABLE_STATISTICS}")
message(STATUS "")
message(STATUS "Backend Support:")
message(STATUS "  OpenBLAS: ${ENABLE_OPENBLAS}")
message(STATUS "  BLIS: ${ENABLE_BLIS}")
message(STATUS "  Intel MKL: ${ENABLE_MKL}")
message(STATUS "  NVIDIA cuBLAS: ${ENABLE_CUBLAS}")
message(STATUS "  AMD rocBLAS: ${ENABLE_ROCBLAS}")
message(STATUS "  Apple Accelerate: ${ENABLE_ACCELERATE}")
message(STATUS "===========================================")
message(STATUS "")
