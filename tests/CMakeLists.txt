cmake_minimum_required(VERSION 3.18)
project(faster-blaster-backend-tests CXX C)

# Enable languages
enable_language(C CXX)
# Note: CUDA language not needed for tests (CUDAToolkit headers/libs are sufficient)
# We don't compile .cu files in tests, only link against cuBLAS libraries

# Set standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 20)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Enable testing framework
enable_testing()

# ============================================================================
# Plugin Architecture Test
# ============================================================================
add_executable(test_plugin_architecture
    test_plugin_architecture.c
)

target_link_libraries(test_plugin_architecture faster-blaster)

add_test(NAME plugin_architecture COMMAND test_plugin_architecture)

# ============================================================================
# Benchmark System Test (Phase 3.0)
# ============================================================================
add_executable(test_benchmark_system
    test_benchmark_system.c
)

target_link_libraries(test_benchmark_system faster-blaster)

add_test(NAME benchmark_system COMMAND test_benchmark_system)

# Minimal benchmark test (no library dependencies)
add_executable(test_benchmark_minimal
    test_benchmark_minimal.c
)

add_test(NAME benchmark_minimal COMMAND test_benchmark_minimal)

# ============================================================================
# Unified Device API Test
# ============================================================================
add_executable(test_device_api
    test_device_api.c
)

target_link_libraries(test_device_api faster-blaster)

add_test(NAME device_api COMMAND test_device_api)

# ============================================================================
# Device Memory Manager Test (No GPU Hardware Required)
# ============================================================================
add_executable(test_device_memory_simple
    test_device_memory_simple.c
)

target_link_libraries(test_device_memory_simple faster-blaster)

# Platform-specific math library
if(UNIX)
    target_link_libraries(test_device_memory_simple m)
endif()

# Copy faster-blaster DLL to test directory
if(WIN32)
    add_custom_command(TARGET test_device_memory_simple POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_device_memory_simple>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME device_memory_simple COMMAND test_device_memory_simple)

message(STATUS "Device Memory Manager simple test enabled")

# ============================================================================
# Smart Wrappers Test (GPU Hardware - Requires CUDA)
# ============================================================================
if(CMAKE_CUDA_COMPILER OR CUDAToolkit_FOUND)
    add_executable(test_smart_wrappers
        test_smart_wrappers.c
    )

    target_link_libraries(test_smart_wrappers faster-blaster)

    # Link CUDA runtime
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_smart_wrappers CUDA::cudart CUDA::cublas)
    endif()

    # Platform-specific math library
    if(UNIX)
        target_link_libraries(test_smart_wrappers m)
    endif()

    # Copy faster-blaster DLL to test directory
    if(WIN32)
        add_custom_command(TARGET test_smart_wrappers POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:faster-blaster>
                $<TARGET_FILE_DIR:test_smart_wrappers>
            COMMENT "Copying faster-blaster.dll to test directory"
        )
    endif()

    add_test(NAME smart_wrappers COMMAND test_smart_wrappers)

    message(STATUS "Smart Wrappers GPU test enabled")
    
    # Additional focused test for Level 1 operations
    add_executable(test_level1_ops
        test_level1_ops.c
    )
    
    target_link_libraries(test_level1_ops faster-blaster)
    
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_level1_ops CUDA::cudart CUDA::cublas)
    endif()
    
    if(UNIX)
        target_link_libraries(test_level1_ops m)
    endif()
    
    if(WIN32)
        add_custom_command(TARGET test_level1_ops POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:faster-blaster>
                $<TARGET_FILE_DIR:test_level1_ops>
            COMMENT "Copying faster-blaster.dll"
        )
    endif()
    
    # Phase 1 GPU operations comprehensive test
    add_executable(test_phase1_gpu_operations
        test_phase1_gpu_operations.c
    )
    
    target_link_libraries(test_phase1_gpu_operations faster-blaster)
    
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_phase1_gpu_operations CUDA::cudart CUDA::cublas)
    endif()
    
    if(UNIX)
        target_link_libraries(test_phase1_gpu_operations m)
    endif()
    
    if(WIN32)
        add_custom_command(TARGET test_phase1_gpu_operations POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:faster-blaster>
                $<TARGET_FILE_DIR:test_phase1_gpu_operations>
            COMMENT "Copying faster-blaster.dll"
        )
    endif()
    
    add_test(NAME phase1_gpu_operations COMMAND test_phase1_gpu_operations)
    # KMP_DUPLICATE_LIB_OK suppresses abort() from dual OpenMP runtime conflict
    # (libiomp5md.dll from AOCL + libomp.dll from LLVM both loaded)
    set_tests_properties(phase1_gpu_operations PROPERTIES
        ENVIRONMENT "KMP_DUPLICATE_LIB_OK=TRUE")
    
    message(STATUS "Phase 1 GPU Operations test enabled")
    
else()
    message(STATUS "Smart Wrappers GPU test disabled (CUDA not available)")
endif()

# ============================================================================
# Comprehensive Correctness Test
# ============================================================================
add_executable(test_comprehensive_correctness
    test_comprehensive_correctness.c
)

target_link_libraries(test_comprehensive_correctness faster-blaster)

# Platform-specific math library
if(UNIX)
    target_link_libraries(test_comprehensive_correctness m)
endif()

# Copy faster-blaster DLL to test directory
if(WIN32)
    add_custom_command(TARGET test_comprehensive_correctness POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_comprehensive_correctness>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME comprehensive_correctness COMMAND test_comprehensive_correctness)

message(STATUS "Comprehensive correctness test enabled")

# ============================================================================
# Reference BLAS Level 1 Expanded Test (Standalone)
# ============================================================================
add_executable(test_blas_level1_reference_expanded
    test_blas_level1_reference_expanded.c
)

# Platform-specific math library
if(UNIX)
    target_link_libraries(test_blas_level1_reference_expanded m)
endif()

add_test(NAME blas_level1_reference_expanded COMMAND test_blas_level1_reference_expanded)

message(STATUS "Reference BLAS Level 1 expanded test enabled")

# ============================================================================
# Judge Module Tests (corpus, store, selection, dispatch)
# ============================================================================
add_executable(test_judge
    test_judge.c
)

target_link_libraries(test_judge faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_judge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_judge>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME judge_module COMMAND test_judge)

message(STATUS "Judge module tests enabled")

# ============================================================================
# Backend Loader Integration Test
# ============================================================================
if(TRUE)
    add_executable(test_backend_loader
        test_backend_loader.c
    )
    
    target_link_libraries(test_backend_loader faster-blaster)
    
    # Platform-specific math library
    if(UNIX)
        target_link_libraries(test_backend_loader m)
    endif()
    
    # Copy faster-blaster DLL to test directory so tests can run from any location
    if(WIN32)
        add_custom_command(TARGET test_backend_loader POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:faster-blaster>
                $<TARGET_FILE_DIR:test_backend_loader>
            COMMENT "Copying faster-blaster.dll to test directory"
        )
    endif()
    
    add_test(NAME backend_loader_integration COMMAND test_backend_loader)
    
    message(STATUS "Backend loader integration test enabled")
endif()

# ============================================================================
# CLBlast Execution Test
# ============================================================================
if(FB_ENABLE_OPENCL)
    add_executable(test_clblast_execution
        test_clblast_execution.c
    )
    
    target_include_directories(test_clblast_execution PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_clblast_execution faster-blaster)
    
    # Platform-specific math library
    if(UNIX)
        target_link_libraries(test_clblast_execution m)
    endif()
    
    # Copy faster-blaster DLL to test directory
    if(WIN32)
        add_custom_command(TARGET test_clblast_execution POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:faster-blaster>
                $<TARGET_FILE_DIR:test_clblast_execution>
            COMMENT "Copying faster-blaster.dll to test directory"
        )
    endif()
    
    add_test(NAME clblast_execution COMMAND test_clblast_execution)
    # Smart wrappers shutdown crash fixed; test now cleans up successfully
    # set_tests_properties(clblast_execution PROPERTIES WILL_FAIL TRUE)
    
    message(STATUS "CLBlast execution test enabled")
endif()

# ============================================================================
# CUDA/cuBLAS Tests
# ============================================================================
if(ENABLE_CUBLAS AND CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    
    # cuBLAS trait implementation
    add_library(cublas_trait STATIC
        ${CMAKE_SOURCE_DIR}/src/backends/gpu/cublas_trait_impl.c
    )
    
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/backends/gpu/cublas_trait_impl.c
        PROPERTIES LANGUAGE CUDA
    )
    
    target_include_directories(cublas_trait PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(cublas_trait
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
    )
    
    # cuBLAS basic test
    add_executable(test_cublas_basic
        test_cublas_basic.c
    )
    
    target_link_libraries(test_cublas_basic
        cublas_trait
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
    )
    
    if(UNIX)
        target_link_libraries(test_cublas_basic m)
    endif()
    
    add_test(NAME CuBLAS_Basic_Tests COMMAND test_cublas_basic)
    
    message(STATUS "cuBLAS tests enabled")
    
    # Device Memory Manager test
    add_executable(test_device_memory_manager
        test_device_memory_manager.c
    )
    
    target_link_libraries(test_device_memory_manager
        faster-blaster
        cublas_trait
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
    )
    
    if(UNIX)
        target_link_libraries(test_device_memory_manager m)
    endif()
    
    # Copy faster-blaster DLL to test directory
    if(WIN32)
        add_custom_command(TARGET test_device_memory_manager POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:faster-blaster>
                $<TARGET_FILE_DIR:test_device_memory_manager>
            COMMENT "Copying faster-blaster.dll to test directory"
        )
    endif()
    
    add_test(NAME Device_Memory_Manager COMMAND test_device_memory_manager)
    
    message(STATUS "Device Memory Manager test enabled")
endif()

# ============================================================================
# AOCL Tests (AMD CPU)
# ============================================================================
if(ENABLE_AOCL)
    # AOCL backend library
    add_library(aocl_backend_lib STATIC
        ${CMAKE_SOURCE_DIR}/src/backends/aocl_backend.c
    )
    
    target_include_directories(aocl_backend_lib PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    # AOCL test executable
    add_executable(test_aocl_backend
        test_aocl_backend.c
    )
    
    target_link_libraries(test_aocl_backend
        aocl_backend_lib
    )
    
    if(UNIX)
        target_link_libraries(test_aocl_backend m dl pthread)
    else()
        target_link_libraries(test_aocl_backend)
    endif()
    
    add_test(NAME AOCL_Backend_Tests COMMAND test_aocl_backend)
    
    message(STATUS "AOCL tests enabled")
endif()

# ============================================================================
# BLIS Tests (Portable CPU)
# ============================================================================
if(ENABLE_BLIS)
    # BLIS backend library
    add_library(blis_backend_lib STATIC
        ${CMAKE_SOURCE_DIR}/src/backends/blis_backend.c
    )
    
    target_include_directories(blis_backend_lib PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    # BLIS test executable
    add_executable(test_blis_backend
        test_blis_backend.c
    )
    
    target_link_libraries(test_blis_backend
        blis_backend_lib
    )
    
    if(UNIX)
        target_link_libraries(test_blis_backend m dl pthread)
    else()
        target_link_libraries(test_blis_backend)
    endif()
    
    add_test(NAME BLIS_Backend_Tests COMMAND test_blis_backend)
    # FIXED: Level 2/3 BLAS wrappers (sgemv, dgemv, sgemm, dgemm) now use CBLAS
    # as the primary path (stable API, plain int params, no LP64/LLP64 ABI issues).
    # All 9 BLIS backend tests pass on Windows.
    
    message(STATUS "BLIS tests enabled")
endif()

# ============================================================================
# oneMKL Tests (Intel GPU / CUDA backend)
# ============================================================================
if(ENABLE_ONEMKL)
    find_package(CUDAToolkit)
    
    # Use pre-built oneMKL trait library (built separately with Intel DPC++ compiler)
    # Build it using: src\backends\gpu\build_onemkl.bat
    set(ONEMKL_TRAIT_LIB "${CMAKE_SOURCE_DIR}/src/backends/gpu/build/onemkl_trait.lib")
    
    if(NOT EXISTS ${ONEMKL_TRAIT_LIB})
        message(FATAL_ERROR "oneMKL trait library not found at ${ONEMKL_TRAIT_LIB}\n"
                            "Please build it first: src\\backends\\gpu\\build_onemkl.bat")
    endif()
    
    # Create an IMPORTED library target
    add_library(onemkl_trait STATIC IMPORTED)
    set_target_properties(onemkl_trait PROPERTIES
        IMPORTED_LOCATION ${ONEMKL_TRAIT_LIB}
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include"
    )
    
    # oneMKL test executable
    add_executable(test_onemkl_backend
        test_onemkl_backend.cpp
    )
    
    set_target_properties(test_onemkl_backend PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # Use dynamic CRT (/MD) to match Intel libraries
    if(MSVC)
        target_compile_options(test_onemkl_backend PRIVATE /MD)
    endif()
    
    # Add Intel oneAPI library directories
    if(WIN32)
        set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
        target_link_directories(test_onemkl_backend PRIVATE
            "${ONEAPI_ROOT}/compiler/latest/lib"
            "${ONEAPI_ROOT}/mkl/latest/lib"
            "${ONEAPI_ROOT}/tbb/latest/lib"
        )
    endif()
    
    # Link libraries - Intel libraries FIRST, then our trait library
    target_link_libraries(test_onemkl_backend
        sycl8
        mkl_sycl
        mkl_intel_lp64
        mkl_tbb_thread
        mkl_core
        tbb12
        ${ONEMKL_TRAIT_LIB}
        OpenCL
    )
    
    if(UNIX)
        target_link_libraries(test_onemkl_backend m pthread)
    endif()
    
    add_test(NAME OneMKL_Backend_Tests COMMAND test_onemkl_backend)
    
    message(STATUS "oneMKL tests enabled")
endif()

# ============================================================================
# MKL CPU Tests
# ============================================================================
if(ENABLE_MKL)
    # MKL backend library
    add_library(mkl_backend_lib STATIC
        ${CMAKE_SOURCE_DIR}/src/backends/mkl_backend.c
    )

    target_include_directories(mkl_backend_lib PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    # MKL test executable
    add_executable(test_mkl_backend test_mkl_backend.c)

    target_link_libraries(test_mkl_backend mkl_backend_lib)

    if(UNIX)
        target_link_libraries(test_mkl_backend m dl pthread)
    endif()

    add_test(NAME MKL_Backend_Tests COMMAND test_mkl_backend)
    message(STATUS "MKL CPU tests enabled")
endif()

# ============================================================================
# OpenBLAS CPU Tests
# ============================================================================
if(ENABLE_OPENBLAS)
    # OpenBLAS backend library
    add_library(openblas_backend_lib STATIC
        ${CMAKE_SOURCE_DIR}/src/backends/openblas_backend.c
    )

    target_include_directories(openblas_backend_lib PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    # OpenBLAS test executable
    add_executable(test_openblas_backend test_openblas_backend.c)

    target_link_libraries(test_openblas_backend openblas_backend_lib)

    if(UNIX)
        target_link_libraries(test_openblas_backend m dl pthread)
    endif()

    add_test(NAME OpenBLAS_Backend_Tests COMMAND test_openblas_backend)
    message(STATUS "OpenBLAS CPU tests enabled")
endif()

# ============================================================================
# OxiBLAS CPU Tests (pure-Rust BLAS/LAPACK)
# ============================================================================
if(ENABLE_OXIBLAS)
    # Minimal backend adapter library (avoids pulling in entire plugin infra)
    add_library(oxiblas_backend_lib STATIC
        ${CMAKE_SOURCE_DIR}/src/backends/oxiblas_backend.c
    )

    target_include_directories(oxiblas_backend_lib PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    if(OXIBLAS_CPU_AVAILABLE)
        target_link_libraries(oxiblas_backend_lib PRIVATE ${OXIBLAS_LIBRARIES})
    endif()

    # OxiBLAS test executable
    add_executable(test_oxiblas_backend test_oxiblas_backend.c)
    target_include_directories(test_oxiblas_backend PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_oxiblas_backend oxiblas_backend_lib)

    if(UNIX)
        target_link_libraries(test_oxiblas_backend m dl pthread)
    endif()

    add_test(NAME OxiBLAS_Backend_Tests COMMAND test_oxiblas_backend)
    message(STATUS "OxiBLAS CPU tests enabled")
endif()

# ============================================================================
# rocBLAS GPU Tests (AMD) - Requires ROCm installed
# ============================================================================
# TODO: Re-enable when rocBLAS backend is fixed (currently disabled due to incomplete CUDA->HIP conversion)
if(FALSE) # was: if(ENABLE_ROCBLAS)
    add_executable(test_rocblas_backend test_rocblas_backend.cpp)
    set_target_properties(test_rocblas_backend PROPERTIES CXX_STANDARD 17)
    target_include_directories(test_rocblas_backend PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_rocblas_backend faster-blaster)
    add_test(NAME rocBLAS_Backend_Tests COMMAND test_rocblas_backend)
    message(STATUS "rocBLAS tests enabled")
endif()

# ============================================================================
# hipBLAS GPU Tests (AMD/NVIDIA portable) - Requires HIP SDK
# ============================================================================
if(ENABLE_HIPBLAS AND ROCM_AVAILABLE AND HIP_COMPILER_AVAILABLE
   AND EXISTS "${CMAKE_SOURCE_DIR}/src/backends/gpu/hipblas_trait_impl.cpp")
    add_executable(test_hipblas_backend test_hipblas_backend.cpp)
    set_target_properties(test_hipblas_backend PROPERTIES CXX_STANDARD 17)
    target_include_directories(test_hipblas_backend PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_hipblas_backend faster-blaster)
    add_test(NAME hipBLAS_Backend_Tests COMMAND test_hipblas_backend)
    message(STATUS "hipBLAS tests enabled")
endif()

# ============================================================================
# Ranked Dispatch Table Tests (Phase 3)
# ============================================================================
add_executable(test_ranked_dispatch
    test_ranked_dispatch.c
)

target_link_libraries(test_ranked_dispatch faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_ranked_dispatch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_ranked_dispatch>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME ranked_dispatch_tables COMMAND test_ranked_dispatch)

message(STATUS "Ranked dispatch table tests enabled")

# ============================================================================
# Constraint-Based Backend Selection Tests (Phase 3)
# ============================================================================
add_executable(test_constraint_selection
    test_constraint_selection.c
)

target_link_libraries(test_constraint_selection faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_constraint_selection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_constraint_selection>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME constraint_based_selection COMMAND test_constraint_selection)

message(STATUS "Constraint-based selection tests enabled")

# ============================================================================
# Operation Sequence Executor Tests (Phase 3)
# ============================================================================
add_executable(test_op_sequence
    test_op_sequence.c
)

target_link_libraries(test_op_sequence faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_op_sequence POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_op_sequence>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME op_sequence_executor COMMAND test_op_sequence)

message(STATUS "Operation sequence executor tests enabled")

# ============================================================================
# Test Summary
# ============================================================================
message(STATUS "")
message(STATUS "Backend Tests Configuration:")
message(STATUS "  cuBLAS: ${ENABLE_CUBLAS}")
message(STATUS "  rocBLAS: ${ENABLE_ROCBLAS}")
message(STATUS "  hipBLAS: ${ENABLE_HIPBLAS}")
message(STATUS "  oneMKL: ${ENABLE_ONEMKL}")
message(STATUS "  MKL (CPU): ${ENABLE_MKL}")
message(STATUS "  OpenBLAS: ${ENABLE_OPENBLAS}")
message(STATUS "  AOCL (BLIS+libFLAME): ${ENABLE_AOCL}")
message(STATUS "  BLIS: ${ENABLE_BLIS}")
message(STATUS "")


