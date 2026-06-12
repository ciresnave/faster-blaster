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

option(FB_BUILD_CONVENTION_THUNK_TESTS
    "Build the convention-thunk test families" OFF)
option(FB_BUILD_CONVENTION_THUNK_SOLVER_FAMILIES
    "Build solver-style thunk families (GESV, LU, Cholesky, SYSV)" ${FB_BUILD_CONVENTION_THUNK_TESTS})
option(FB_BUILD_CONVENTION_THUNK_FACTORIZATION_FAMILIES
    "Build factorization-style thunk families (QR, Q, apply-Q, bidiag, Hessenberg, tridiagonal)" ${FB_BUILD_CONVENTION_THUNK_TESTS})
option(FB_BUILD_CONVENTION_THUNK_SPECIAL_FAMILIES
    "Build packed and special thunk families" ${FB_BUILD_CONVENTION_THUNK_TESTS})

# ============================================================================
# Plugin Architecture Test
# ============================================================================
add_executable(test_plugin_architecture
    test_plugin_architecture.c
)

target_link_libraries(test_plugin_architecture faster-blaster)

add_test(NAME plugin_architecture COMMAND test_plugin_architecture)

# ============================================================================
# Plugin Selection Order Regression Test
# ============================================================================
add_executable(test_plugin_selection_order
    test_plugin_selection_order.c
)

target_link_libraries(test_plugin_selection_order faster-blaster)

add_test(NAME plugin_selection_order COMMAND test_plugin_selection_order)

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

if(WIN32)
    set_tests_properties(device_api PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:faster-blaster>;PATH=path_list_prepend:${CMAKE_BINARY_DIR}/backends-install/openblas/bin")
endif()

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
# Judge Runner — end-to-end precision + timing CI test
#
# Wires fb_judge_init() + oracle (reference) + candidate backends +
# fb_judge_run() into a single executable.  Phase 1 covers all 20 currently
# implemented DIRECT/INDEX operations against the built-in reference backend.
# Returns non-zero if any mandatory operation falls below its digit threshold.
# ============================================================================
add_executable(test_judge_runner
    test_judge_runner.c
)

target_include_directories(test_judge_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(test_judge_runner PRIVATE
    FB_REFERENCE_DLL_DIR="${CMAKE_SOURCE_DIR}/../faster-blaster-reference/build-extended"
    FB_REFERENCE_OP_LIST_PATH="${CMAKE_SOURCE_DIR}/../faster-blaster-reference/all_operations_in_reference.txt"
)

target_link_libraries(test_judge_runner faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_judge_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_judge_runner>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME judge_end_to_end COMMAND test_judge_runner)
# KMP_DUPLICATE_LIB_OK avoids abort() when both libiomp5md and libomp are loaded
# (can happen if AOCL or MKL backends are later added to k_backends[]).
set_tests_properties(judge_end_to_end PROPERTIES
    ENVIRONMENT "KMP_DUPLICATE_LIB_OK=TRUE")

message(STATUS "Judge end-to-end runner test enabled")

# ============================================================================
# Reference Loader Audit — full 3054-op namespace visibility check
# ============================================================================
add_executable(test_reference_loader_audit
    test_reference_loader_audit.c
)

target_include_directories(test_reference_loader_audit PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(test_reference_loader_audit PRIVATE
    FB_REFERENCE_DLL_DIR="${CMAKE_SOURCE_DIR}/../faster-blaster-reference/build-extended"
)

target_link_libraries(test_reference_loader_audit faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_reference_loader_audit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_reference_loader_audit>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME reference_loader_audit COMMAND test_reference_loader_audit)

message(STATUS "Reference loader audit test enabled")

# ============================================================================
# Reference Loader Alias Regression — plain stem + underscore twin must not
# hydrate a raw CBLAS slot with Fortran ABI.
# ============================================================================
add_executable(test_reference_loader_alias_regression
    test_reference_loader_alias_regression.c
)

target_include_directories(test_reference_loader_alias_regression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(test_reference_loader_alias_regression PRIVATE
    FB_REFERENCE_DLL_DIR="${CMAKE_SOURCE_DIR}/../faster-blaster-reference/build-extended"
)

target_link_libraries(test_reference_loader_alias_regression faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_reference_loader_alias_regression POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_reference_loader_alias_regression>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME reference_loader_alias_regression COMMAND test_reference_loader_alias_regression)

message(STATUS "Reference loader alias regression test enabled")

# ============================================================================
# Convention Thunks — opt-in synthetic coverage for thunk-heavy delegation
# paths. The umbrella switch seeds the subgroup defaults; individual groups can
# still be toggled independently for narrower profiling.
# ============================================================================

# ----------------------------------------------------------------------------
# Solver families
# ----------------------------------------------------------------------------
if(FB_BUILD_CONVENTION_THUNK_SOLVER_FAMILIES)

# ============================================================================
# GESV Convention Thunks — direct synthetic coverage for both Fortran->CBLAS
# and CBLAS->Fortran delegation paths.
# ============================================================================
add_executable(test_conv_thunks_gesv
    test_conv_thunks_gesv.c
)

target_include_directories(test_conv_thunks_gesv PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gesv faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gesv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gesv>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gesv COMMAND test_conv_thunks_gesv)

message(STATUS "GESV convention thunk test enabled")

# ============================================================================
# LU Convention Thunks — direct synthetic coverage for GETRF and GETRS
# delegation and transpose preservation.
# ============================================================================
add_executable(test_conv_thunks_lu_family
    test_conv_thunks_lu_family.c
)

target_include_directories(test_conv_thunks_lu_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lu_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lu_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lu_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lu_family COMMAND test_conv_thunks_lu_family)

message(STATUS "LU convention thunk test enabled")

# ============================================================================
# Cholesky Convention Thunks — direct synthetic coverage for POTRF, POTRS,
# POTRI, and POSV, including row-major RHS translation.
# ============================================================================
add_executable(test_conv_thunks_cholesky_family
    test_conv_thunks_cholesky_family.c
)

target_include_directories(test_conv_thunks_cholesky_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_cholesky_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_cholesky_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_cholesky_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_cholesky_family COMMAND test_conv_thunks_cholesky_family)

message(STATUS "Cholesky convention thunk test enabled")

# ============================================================================
# Pivoted Symmetric/Hermitian Solve Convention Thunks — direct synthetic
# coverage for SYSV/HESV workspace-query bridging and row-major translation.
# ============================================================================
add_executable(test_conv_thunks_sysv_family
    test_conv_thunks_sysv_family.c
)

target_include_directories(test_conv_thunks_sysv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sysv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sysv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sysv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sysv_family COMMAND test_conv_thunks_sysv_family)

message(STATUS "SYSV/HESV convention thunk test enabled")

endif()

# ----------------------------------------------------------------------------
# Factorization families
# ----------------------------------------------------------------------------
if(FB_BUILD_CONVENTION_THUNK_FACTORIZATION_FAMILIES)

# ============================================================================
# QR-Like Factorization Convention Thunks — direct synthetic coverage for
# GEQRF/GELQF/GEQLF/GERQF workspace-query bridging and row-major translation.
# ============================================================================
add_executable(test_conv_thunks_qr_family
    test_conv_thunks_qr_family.c
)

target_include_directories(test_conv_thunks_qr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_qr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_qr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_qr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_qr_family COMMAND test_conv_thunks_qr_family)

message(STATUS "QR-like convention thunk test enabled")

# ============================================================================
# Explicit Q-Generator Convention Thunks — direct synthetic coverage for
# ORGQR/UNGQR-style workspace-query bridging and row-major translation.
# ============================================================================
add_executable(test_conv_thunks_q_generator_family
    test_conv_thunks_q_generator_family.c
)

target_include_directories(test_conv_thunks_q_generator_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_q_generator_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_q_generator_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_q_generator_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_q_generator_family COMMAND test_conv_thunks_q_generator_family)

message(STATUS "Explicit Q-generator convention thunk test enabled")

# ============================================================================
# Reflector-application Convention Thunks — direct synthetic coverage for
# ORMQR/UNMQR-style workspace-query bridging and row-major A/C translation.
# ============================================================================
add_executable(test_conv_thunks_apply_q_family
    test_conv_thunks_apply_q_family.c
)

target_include_directories(test_conv_thunks_apply_q_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_apply_q_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_apply_q_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_apply_q_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_apply_q_family COMMAND test_conv_thunks_apply_q_family)

message(STATUS "Reflector-application convention thunk test enabled")

# ============================================================================
# Bidiagonal Generator/Application Convention Thunks — direct synthetic
# coverage for ORGBR/UNGBR and ORMBR/UNMBR vect-aware workspace-query bridges.
# ============================================================================
add_executable(test_conv_thunks_bidiag_family
    test_conv_thunks_bidiag_family.c
)

target_include_directories(test_conv_thunks_bidiag_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_bidiag_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_bidiag_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_bidiag_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_bidiag_family COMMAND test_conv_thunks_bidiag_family)

message(STATUS "Bidiagonal generator/application convention thunk test enabled")

# ============================================================================
# Hessenberg Reflector-Application Convention Thunks — direct synthetic
# coverage for ORMHR/UNMHR workspace-query bridges with column-major-only
# layout policy.
# ============================================================================
add_executable(test_conv_thunks_hessenberg_apply_family
    test_conv_thunks_hessenberg_apply_family.c
)

target_include_directories(test_conv_thunks_hessenberg_apply_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hessenberg_apply_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hessenberg_apply_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hessenberg_apply_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hessenberg_apply_family COMMAND test_conv_thunks_hessenberg_apply_family)

message(STATUS "Hessenberg reflector-application convention thunk test enabled")

# ============================================================================
# Hessenberg Generator Convention Thunks — direct synthetic coverage for
# ORGHR/UNGHR workspace-query bridges and row-major matrix translation.
# ============================================================================
add_executable(test_conv_thunks_hessenberg_generator_family
    test_conv_thunks_hessenberg_generator_family.c
)

target_include_directories(test_conv_thunks_hessenberg_generator_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hessenberg_generator_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hessenberg_generator_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hessenberg_generator_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hessenberg_generator_family COMMAND test_conv_thunks_hessenberg_generator_family)

message(STATUS "Hessenberg generator convention thunk test enabled")

# ============================================================================
# Tridiagonal-Reduction Generator Convention Thunks — direct synthetic
# coverage for ORGTR/UNGTR workspace-query bridges with UPLO remapping and
# row-major matrix translation.
# ============================================================================
add_executable(test_conv_thunks_tridiag_generator_family
    test_conv_thunks_tridiag_generator_family.c
)

target_include_directories(test_conv_thunks_tridiag_generator_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_tridiag_generator_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_tridiag_generator_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_tridiag_generator_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_tridiag_generator_family COMMAND test_conv_thunks_tridiag_generator_family)

message(STATUS "Tridiagonal-reduction generator convention thunk test enabled")

# ============================================================================
# Unblocked Q-Generator Convention Thunks — direct synthetic coverage for
# ORG2L/UNG2L bridges with row-major matrix translation and internal scratch.
# ============================================================================
add_executable(test_conv_thunks_unblocked_q_generator_family
    test_conv_thunks_unblocked_q_generator_family.c
)

target_include_directories(test_conv_thunks_unblocked_q_generator_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_unblocked_q_generator_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_unblocked_q_generator_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_unblocked_q_generator_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_unblocked_q_generator_family COMMAND test_conv_thunks_unblocked_q_generator_family)

message(STATUS "Unblocked Q-generator convention thunk test enabled")

# ============================================================================
# Unblocked Matrix+TAU Factorization Convention Thunks — direct synthetic
# coverage for GELQ2/GEQL2/GERQ2 bridges with row-major matrix translation.
# ============================================================================
add_executable(test_conv_thunks_unblocked_matrix_tau_factor_family
    test_conv_thunks_unblocked_matrix_tau_factor_family.c
)

target_include_directories(test_conv_thunks_unblocked_matrix_tau_factor_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_unblocked_matrix_tau_factor_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_unblocked_matrix_tau_factor_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_unblocked_matrix_tau_factor_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_unblocked_matrix_tau_factor_family COMMAND test_conv_thunks_unblocked_matrix_tau_factor_family)

message(STATUS "Unblocked matrix+tau factorization convention thunk test enabled")

endif()

# ----------------------------------------------------------------------------
# Special families
# ----------------------------------------------------------------------------
if(FB_BUILD_CONVENTION_THUNK_SPECIAL_FAMILIES)

# ============================================================================
# LAPACK LATRZ Convention Thunks — direct synthetic coverage for
# SLATRZ/DLATRZ/CLATRZ/ZLATRZ bridges with row-major matrix translation and
# no-info Fortran ABI delegation.
# ============================================================================
add_executable(test_conv_thunks_latrz_family
    test_conv_thunks_latrz_family.c
)

target_include_directories(test_conv_thunks_latrz_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_latrz_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_latrz_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_latrz_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_latrz_family COMMAND test_conv_thunks_latrz_family)

message(STATUS "LAPACK LATRZ convention thunk test enabled")

# ============================================================================
# LAPACK TZRZF Convention Thunks — focused synthetic coverage for
# STZRZF/DTZRZF/CTZRZF/ZTZRZF workspace-query bridges.
# ============================================================================
add_executable(test_conv_thunks_tzrzf_family
    test_conv_thunks_tzrzf_family.c
)

target_include_directories(test_conv_thunks_tzrzf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_tzrzf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_tzrzf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_tzrzf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_tzrzf_family COMMAND test_conv_thunks_tzrzf_family)

message(STATUS "LAPACK TZRZF convention thunk test enabled")

# ============================================================================
# LAPACK LAUU2/LAUUM Convention Thunks — focused synthetic coverage for
# in-place triangular product bridges and row-major uplo flipping.
# ============================================================================
add_executable(test_conv_thunks_lauu_family
    test_conv_thunks_lauu_family.c
)

target_include_directories(test_conv_thunks_lauu_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lauu_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lauu_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lauu_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lauu_family COMMAND test_conv_thunks_lauu_family)

message(STATUS "LAPACK LAUU2/LAUUM convention thunk test enabled")

# ============================================================================
# LAPACK LATRD Convention Thunks — focused synthetic coverage for
# UPLO-driven tridiagonal-panel reduction bridges with A/W matrix copies.
# ============================================================================
add_executable(test_conv_thunks_latrd_family
    test_conv_thunks_latrd_family.c
)

target_include_directories(test_conv_thunks_latrd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_latrd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_latrd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_latrd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_latrd_family COMMAND test_conv_thunks_latrd_family)

message(STATUS "LAPACK LATRD convention thunk test enabled")

# ============================================================================
# LAPACK LATDF Convention Thunks — focused synthetic coverage for
# no-info LU-factorization estimate bridges with row-major Z copies.
# ============================================================================
add_executable(test_conv_thunks_latdf_family
    test_conv_thunks_latdf_family.c
)

target_include_directories(test_conv_thunks_latdf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_latdf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_latdf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_latdf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_latdf_family COMMAND test_conv_thunks_latdf_family)

message(STATUS "LAPACK LATDF convention thunk test enabled")

# ============================================================================
# LAPACK LATRS Convention Thunks — focused synthetic coverage for
# triangular solve bridges with row-major A copies and normin/cnorm flow.
# ============================================================================
add_executable(test_conv_thunks_latrs_family
    test_conv_thunks_latrs_family.c
)

target_include_directories(test_conv_thunks_latrs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_latrs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_latrs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_latrs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_latrs_family COMMAND test_conv_thunks_latrs_family)

message(STATUS "LAPACK LATRS convention thunk test enabled")

# ============================================================================
# LAPACK LATPS Convention Thunks — focused synthetic coverage for
# packed triangular solve bridges with row-major packed storage translation.
# ============================================================================
add_executable(test_conv_thunks_latps_family
    test_conv_thunks_latps_family.c
)

target_include_directories(test_conv_thunks_latps_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_latps_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_latps_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_latps_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_latps_family COMMAND test_conv_thunks_latps_family)

message(STATUS "LAPACK LATPS convention thunk test enabled")

# ============================================================================
# LAPACK LATBS Convention Thunks — focused synthetic coverage for
# triangular band solve bridges with row-major compact band translation.
# ============================================================================
add_executable(test_conv_thunks_latbs_family
    test_conv_thunks_latbs_family.c
)

target_include_directories(test_conv_thunks_latbs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_latbs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_latbs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_latbs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_latbs_family COMMAND test_conv_thunks_latbs_family)

message(STATUS "LAPACK LATBS convention thunk test enabled")

# ============================================================================
# LAPACK LASSQ Convention Thunks — focused synthetic coverage for
# scale/sumsq vector-reduction bridges with no layout translation.
# ============================================================================
add_executable(test_conv_thunks_lassq_family
    test_conv_thunks_lassq_family.c
)

target_include_directories(test_conv_thunks_lassq_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lassq_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lassq_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lassq_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lassq_family COMMAND test_conv_thunks_lassq_family)

message(STATUS "LAPACK LASSQ convention thunk test enabled")

# ============================================================================
# LAPACK LAEV2 Convention Thunks — focused synthetic coverage for
# scalar eigensystem bridges with no layout translation or info slot.
# ============================================================================
add_executable(test_conv_thunks_laev2_family
    test_conv_thunks_laev2_family.c
)

target_include_directories(test_conv_thunks_laev2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laev2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laev2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laev2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laev2_family COMMAND test_conv_thunks_laev2_family)

message(STATUS "LAPACK LAEV2 convention thunk test enabled")

# ============================================================================
# LAPACK LASV2 Convention Thunks — focused synthetic coverage for
# scalar 2x2 SVD bridges with no layout translation or info slot.
# ============================================================================
add_executable(test_conv_thunks_lasv2_family
    test_conv_thunks_lasv2_family.c
)

target_include_directories(test_conv_thunks_lasv2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasv2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasv2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasv2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasv2_family COMMAND test_conv_thunks_lasv2_family)

message(STATUS "LAPACK LASV2 convention thunk test enabled")

# ============================================================================
# LAPACK LA_GBAMV Convention Thunks — focused synthetic coverage for
# absolute band-matrix vector helpers with real and complex payload splits.
# ============================================================================
add_executable(test_conv_thunks_la_gbamv_family
    test_conv_thunks_la_gbamv_family.c
)

target_include_directories(test_conv_thunks_la_gbamv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_la_gbamv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_la_gbamv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_la_gbamv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_la_gbamv_family COMMAND test_conv_thunks_la_gbamv_family)

message(STATUS "LAPACK LA_GBAMV convention thunk test enabled")

# ============================================================================
# CNRM2/ZNRM2 Convention Thunks — focused synthetic coverage for
# complex norm return-value alias bridges.
# ============================================================================
add_executable(test_conv_thunks_cnrm2_znrm2_family
    test_conv_thunks_cnrm2_znrm2_family.c
)

target_include_directories(test_conv_thunks_cnrm2_znrm2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_cnrm2_znrm2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_cnrm2_znrm2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_cnrm2_znrm2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_cnrm2_znrm2_family COMMAND test_conv_thunks_cnrm2_znrm2_family)

message(STATUS "CNRM2/ZNRM2 convention thunk test enabled")

# ============================================================================
# LAPACK PPCON Convention Thunks — focused synthetic coverage for
# packed positive-definite condition-estimate helpers.
# ============================================================================
add_executable(test_conv_thunks_ppcon_family
    test_conv_thunks_ppcon_family.c
)

target_include_directories(test_conv_thunks_ppcon_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ppcon_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ppcon_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ppcon_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ppcon_family COMMAND test_conv_thunks_ppcon_family)

message(STATUS "LAPACK PPCON convention thunk test enabled")

# ============================================================================
# Symmetric packed condition convention thunks — focused synthetic coverage for
# SSPCON/DSPCON/CSPCON/ZSPCON with packed-storage reorder, pivot forwarding,
# and real-vs-complex hidden scratch differences.
# ============================================================================
add_executable(test_conv_thunks_spcon_family
    test_conv_thunks_spcon_family.c
)

target_include_directories(test_conv_thunks_spcon_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spcon_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spcon_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spcon_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spcon_family COMMAND test_conv_thunks_spcon_family)

message(STATUS "Symmetric packed condition convention thunk test enabled")

# ============================================================================
# Hermitian packed condition convention thunks — focused synthetic coverage for
# CHPCON/ZHPCON with packed-storage reorder, pivot forwarding, hidden WORK(2*n),
# and compatibility RWORK forwarding for the reference ABI.
# ============================================================================
add_executable(test_conv_thunks_hpcon_family
    test_conv_thunks_hpcon_family.c
)

target_include_directories(test_conv_thunks_hpcon_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpcon_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpcon_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpcon_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpcon_family COMMAND test_conv_thunks_hpcon_family)

message(STATUS "Hermitian packed condition convention thunk test enabled")

# ============================================================================
# LAPACK PPEQU Convention Thunks — focused synthetic coverage for
# packed positive-definite equilibration helpers.
# ============================================================================
add_executable(test_conv_thunks_ppequ_family
    test_conv_thunks_ppequ_family.c
)

target_include_directories(test_conv_thunks_ppequ_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ppequ_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ppequ_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ppequ_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ppequ_family COMMAND test_conv_thunks_ppequ_family)

message(STATUS "LAPACK PPEQU convention thunk test enabled")

# ============================================================================
# LAPACK PPRFS Convention Thunks — focused synthetic coverage for
# packed positive-definite iterative-refinement helpers.
# ============================================================================
add_executable(test_conv_thunks_pprfs_family
    test_conv_thunks_pprfs_family.c
)

target_include_directories(test_conv_thunks_pprfs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pprfs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pprfs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pprfs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pprfs_family COMMAND test_conv_thunks_pprfs_family)

message(STATUS "LAPACK PPRFS convention thunk test enabled")

# ============================================================================
# LAPACK HPRFS Convention Thunks — focused synthetic coverage for packed
# Hermitian iterative-refinement helpers with pivot forwarding and row-major
# packed-storage translation.
# ============================================================================
add_executable(test_conv_thunks_hprfs_family
    test_conv_thunks_hprfs_family.c
)

target_include_directories(test_conv_thunks_hprfs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hprfs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hprfs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hprfs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hprfs_family COMMAND test_conv_thunks_hprfs_family)

message(STATUS "LAPACK HPRFS convention thunk test enabled")

# ============================================================================
# LAPACK PPSV Convention Thunks — focused synthetic coverage for
# packed positive-definite direct solve helpers.
# ============================================================================
add_executable(test_conv_thunks_ppsv_family
    test_conv_thunks_ppsv_family.c
)

target_include_directories(test_conv_thunks_ppsv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ppsv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ppsv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ppsv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ppsv_family COMMAND test_conv_thunks_ppsv_family)

message(STATUS "LAPACK PPSV convention thunk test enabled")

# ============================================================================
# Hermitian Packed Solve Convention Thunks — focused synthetic coverage for
# CHPSV/ZHPSV with packed-storage reorder, pivot forwarding, and row-major
# copy-back for both AP and B.
# ============================================================================
add_executable(test_conv_thunks_hpsv_family
    test_conv_thunks_hpsv_family.c
)

target_include_directories(test_conv_thunks_hpsv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpsv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpsv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpsv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpsv_family COMMAND test_conv_thunks_hpsv_family)

message(STATUS "LAPACK HPSV convention thunk test enabled")

# ============================================================================
# LAPACK PPSVX Convention Thunks — focused synthetic coverage for
# packed positive-definite expert solve helpers.
# ============================================================================
add_executable(test_conv_thunks_ppsvx_family
    test_conv_thunks_ppsvx_family.c
)

target_include_directories(test_conv_thunks_ppsvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ppsvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ppsvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ppsvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ppsvx_family COMMAND test_conv_thunks_ppsvx_family)

message(STATUS "LAPACK PPSVX convention thunk test enabled")

# ============================================================================
# LAPACK LAEIN Convention Thunks — focused synthetic coverage for
# split real/complex inverse-iteration helpers that return info through C int.
# ============================================================================
add_executable(test_conv_thunks_laein_family
    test_conv_thunks_laein_family.c
)

target_include_directories(test_conv_thunks_laein_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laein_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laein_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laein_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laein_family COMMAND test_conv_thunks_laein_family)

message(STATUS "LAPACK LAEIN convention thunk test enabled")

# ============================================================================
# LAPACK LAHQR Convention Thunks — focused synthetic coverage for
# split real/complex Hessenberg QR helpers that return info through C int.
# ============================================================================
add_executable(test_conv_thunks_lahqr_family
    test_conv_thunks_lahqr_family.c
)

target_include_directories(test_conv_thunks_lahqr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lahqr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lahqr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lahqr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lahqr_family COMMAND test_conv_thunks_lahqr_family)

message(STATUS "LAPACK LAHQR convention thunk test enabled")

# ============================================================================
# LAPACK LAHRD Convention Thunks — focused synthetic coverage for
# split real/complex Hessenberg-reduction helpers with mixed info behavior.
# ============================================================================
add_executable(test_conv_thunks_lahrd_family
    test_conv_thunks_lahrd_family.c
)

target_include_directories(test_conv_thunks_lahrd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lahrd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lahrd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lahrd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lahrd_family COMMAND test_conv_thunks_lahrd_family)

message(STATUS "LAPACK LAHRD convention thunk test enabled")

# ============================================================================
# LAPACK LAQR0 Convention Thunks — focused synthetic coverage for
# real-only workspace-heavy QR helpers that return info through C int.
# ============================================================================
add_executable(test_conv_thunks_laqr0_family
    test_conv_thunks_laqr0_family.c
)

target_include_directories(test_conv_thunks_laqr0_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laqr0_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laqr0_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laqr0_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laqr0_family COMMAND test_conv_thunks_laqr0_family)

message(STATUS "LAPACK LAQR0 convention thunk test enabled")

# ============================================================================
# LAPACK LAQR1 Convention Thunks — focused synthetic coverage for
# real-only no-info shift helpers with scalar-by-value forwarding.
# ============================================================================
add_executable(test_conv_thunks_laqr1_family
    test_conv_thunks_laqr1_family.c
)

target_include_directories(test_conv_thunks_laqr1_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laqr1_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laqr1_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laqr1_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laqr1_family COMMAND test_conv_thunks_laqr1_family)

message(STATUS "LAPACK LAQR1 convention thunk test enabled")

# ============================================================================
# LAPACK LAQR2 Convention Thunks — focused synthetic coverage for
# real-only no-info QR window helpers with integer and workspace outputs.
# ============================================================================
add_executable(test_conv_thunks_laqr2_family
    test_conv_thunks_laqr2_family.c
)

target_include_directories(test_conv_thunks_laqr2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laqr2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laqr2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laqr2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laqr2_family COMMAND test_conv_thunks_laqr2_family)

message(STATUS "LAPACK LAQR2 convention thunk test enabled")

# ============================================================================
# LAPACK LAQP2 Convention Thunks — focused synthetic coverage for
# no-info pivoting helpers with mixed complex and real auxiliary buffers.
# ============================================================================
add_executable(test_conv_thunks_laqp2_family
    test_conv_thunks_laqp2_family.c
)

target_include_directories(test_conv_thunks_laqp2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laqp2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laqp2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laqp2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laqp2_family COMMAND test_conv_thunks_laqp2_family)

message(STATUS "LAPACK LAQP2 convention thunk test enabled")

# ============================================================================
# LAPACK LAQPS Convention Thunks — focused synthetic coverage for
# no-info pivoting helpers with kb output and mixed complex/real auxiliaries.
# ============================================================================
add_executable(test_conv_thunks_laqps_family
    test_conv_thunks_laqps_family.c
)

target_include_directories(test_conv_thunks_laqps_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laqps_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laqps_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laqps_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laqps_family COMMAND test_conv_thunks_laqps_family)

message(STATUS "LAPACK LAQPS convention thunk test enabled")

# ============================================================================
# LAPACK LARNV Convention Thunks — focused synthetic coverage for
# no-info random-vector helpers with in-place seed updates.
# ============================================================================
add_executable(test_conv_thunks_larnv_family
    test_conv_thunks_larnv_family.c
)

target_include_directories(test_conv_thunks_larnv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larnv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larnv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larnv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larnv_family COMMAND test_conv_thunks_larnv_family)

message(STATUS "LAPACK LARNV convention thunk test enabled")

# ============================================================================
# LAPACK LARGE Convention Thunks — focused synthetic coverage for
# real-only seeded similarity helpers that return info through the C int.
# ============================================================================
add_executable(test_conv_thunks_large_family
    test_conv_thunks_large_family.c
)

target_include_directories(test_conv_thunks_large_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_large_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_large_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_large_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_large_family COMMAND test_conv_thunks_large_family)

message(STATUS "LAPACK LARGE convention thunk test enabled")

# ============================================================================
# LAPACK LARZ Convention Thunks — focused synthetic coverage for
# no-info reflector-application helpers with char-by-value C-side dispatch.
# ============================================================================
add_executable(test_conv_thunks_larz_family
    test_conv_thunks_larz_family.c
)

target_include_directories(test_conv_thunks_larz_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larz_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larz_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larz_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larz_family COMMAND test_conv_thunks_larz_family)

message(STATUS "LAPACK LARZ convention thunk test enabled")

# ============================================================================
# LAPACK LARZB Convention Thunks — focused synthetic coverage for
# no-info block-reflector helpers with four char-by-value C-side selectors.
# ============================================================================
add_executable(test_conv_thunks_larzb_family
    test_conv_thunks_larzb_family.c
)

target_include_directories(test_conv_thunks_larzb_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larzb_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larzb_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larzb_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larzb_family COMMAND test_conv_thunks_larzb_family)

message(STATUS "LAPACK LARZB convention thunk test enabled")

# ============================================================================
# LAPACK LARZT Convention Thunks — focused synthetic coverage for
# no-info triangular-factor helpers with two char-by-value C-side selectors.
# ============================================================================
add_executable(test_conv_thunks_larzt_family
    test_conv_thunks_larzt_family.c
)

target_include_directories(test_conv_thunks_larzt_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larzt_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larzt_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larzt_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larzt_family COMMAND test_conv_thunks_larzt_family)

message(STATUS "LAPACK LARZT convention thunk test enabled")

# ============================================================================
# LAPACK LARFB Convention Thunks — focused synthetic coverage for
# layout-aware block reflector application with hidden workspace.
# ============================================================================
add_executable(test_conv_thunks_larfb_family
    test_conv_thunks_larfb_family.c
)

target_include_directories(test_conv_thunks_larfb_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larfb_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larfb_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larfb_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larfb_family COMMAND test_conv_thunks_larfb_family)

message(STATUS "LAPACK LARFB convention thunk test enabled")

# ============================================================================
# LAPACK LARFT Convention Thunks — focused synthetic coverage for
# split real/complex Fortran ABIs feeding one layout-aware C surface.
# ============================================================================
add_executable(test_conv_thunks_larft_family
    test_conv_thunks_larft_family.c
)

target_include_directories(test_conv_thunks_larft_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larft_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larft_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larft_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larft_family COMMAND test_conv_thunks_larft_family)

message(STATUS "LAPACK LARFT convention thunk test enabled")

# ============================================================================
# LAPACK LARFX Convention Thunks — focused synthetic coverage for
# split real/complex Fortran ABIs with a unified layout-aware C surface.
# ============================================================================
add_executable(test_conv_thunks_larfx_family
    test_conv_thunks_larfx_family.c
)

target_include_directories(test_conv_thunks_larfx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larfx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larfx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larfx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larfx_family COMMAND test_conv_thunks_larfx_family)

message(STATUS "LAPACK LARFX convention thunk test enabled")

# ============================================================================
# LAPACK LARGV Convention Thunks — focused synthetic coverage for
# complex vector rotation generation with real cosine outputs.
# ============================================================================
add_executable(test_conv_thunks_largv_family
    test_conv_thunks_largv_family.c
)

target_include_directories(test_conv_thunks_largv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_largv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_largv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_largv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_largv_family COMMAND test_conv_thunks_largv_family)

message(STATUS "LAPACK LARGV convention thunk test enabled")

# ============================================================================
# LAPACK LARRA Convention Thunks — focused synthetic coverage for
# split-point detection helpers with scalar tolerances by value.
# ============================================================================
add_executable(test_conv_thunks_larra_family
    test_conv_thunks_larra_family.c
)

target_include_directories(test_conv_thunks_larra_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larra_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larra_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larra_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larra_family COMMAND test_conv_thunks_larra_family)

message(STATUS "LAPACK LARRA convention thunk test enabled")

# ============================================================================
# LAPACK LARRB Convention Thunks — focused synthetic coverage for
# interval refinement helpers with scalar indices/tolerances by value.
# ============================================================================
add_executable(test_conv_thunks_larrb_family
    test_conv_thunks_larrb_family.c
)

target_include_directories(test_conv_thunks_larrb_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrb_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrb_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrb_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrb_family COMMAND test_conv_thunks_larrb_family)

message(STATUS "LAPACK LARRB convention thunk test enabled")

# ============================================================================
# LAPACK LARRC Convention Thunks — focused synthetic coverage for
# eigenvalue count helpers with scalar bounds by value.
# ============================================================================
add_executable(test_conv_thunks_larrc_family
    test_conv_thunks_larrc_family.c
)

target_include_directories(test_conv_thunks_larrc_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrc_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrc_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrc_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrc_family COMMAND test_conv_thunks_larrc_family)

message(STATUS "LAPACK LARRC convention thunk test enabled")

# ============================================================================
# LAPACK LARRD Convention Thunks — focused synthetic coverage for
# tridiagonal eigenvalue interval selection with mixed scalar I/O.
# ============================================================================
add_executable(test_conv_thunks_larrd_family
    test_conv_thunks_larrd_family.c
)

target_include_directories(test_conv_thunks_larrd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrd_family COMMAND test_conv_thunks_larrd_family)

message(STATUS "LAPACK LARRD convention thunk test enabled")

# ============================================================================
# LAPACK LARRE Convention Thunks — focused synthetic coverage for
# RRR setup helpers with in-out scalar bounds and output pivmin.
# ============================================================================
add_executable(test_conv_thunks_larre_family
    test_conv_thunks_larre_family.c
)

target_include_directories(test_conv_thunks_larre_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larre_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larre_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larre_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larre_family COMMAND test_conv_thunks_larre_family)

message(STATUS "LAPACK LARRE convention thunk test enabled")

# ============================================================================
# LAPACK LARRF Convention Thunks — focused synthetic coverage for
# relative robust representation shifts with sigma output.
# ============================================================================
add_executable(test_conv_thunks_larrf_family
    test_conv_thunks_larrf_family.c
)

target_include_directories(test_conv_thunks_larrf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrf_family COMMAND test_conv_thunks_larrf_family)

message(STATUS "LAPACK LARRF convention thunk test enabled")

# ============================================================================
# LAPACK LARRJ Convention Thunks — focused synthetic coverage for
# bisection refinement with in-out eigenvalue/error arrays.
# ============================================================================
add_executable(test_conv_thunks_larrj_family
    test_conv_thunks_larrj_family.c
)

target_include_directories(test_conv_thunks_larrj_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrj_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrj_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrj_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrj_family COMMAND test_conv_thunks_larrj_family)

message(STATUS "LAPACK LARRJ convention thunk test enabled")

# ============================================================================
# LAPACK LARRK Convention Thunks — focused synthetic coverage for
# single-eigenvalue bisection with scalar bounds and output scalars.
# ============================================================================
add_executable(test_conv_thunks_larrk_family
    test_conv_thunks_larrk_family.c
)

target_include_directories(test_conv_thunks_larrk_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrk_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrk_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrk_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrk_family COMMAND test_conv_thunks_larrk_family)

message(STATUS "LAPACK LARRK convention thunk test enabled")

# ============================================================================
# LAPACK LARRR Convention Thunks — focused synthetic coverage for
# relative-accuracy screening with in-out subdiagonal storage.
# ============================================================================
add_executable(test_conv_thunks_larrr_family
    test_conv_thunks_larrr_family.c
)

target_include_directories(test_conv_thunks_larrr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrr_family COMMAND test_conv_thunks_larrr_family)

message(STATUS "LAPACK LARRR convention thunk test enabled")

endif()

# ============================================================================
# LAPACK LARRV Convention Thunks — focused synthetic coverage for
# tridiagonal eigenvector backsolve helpers with mixed in/out buffers.
# ============================================================================
add_executable(test_conv_thunks_larrv_family
    test_conv_thunks_larrv_family.c
)

target_include_directories(test_conv_thunks_larrv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_larrv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_larrv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_larrv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_larrv_family COMMAND test_conv_thunks_larrv_family)

message(STATUS "LAPACK LARRV convention thunk test enabled")

# ============================================================================
# LAPACK LARTG Convention Thunks — focused synthetic coverage for
# real and complex Givens rotation generators without info returns.
# ============================================================================
add_executable(test_conv_thunks_lartg_family
    test_conv_thunks_lartg_family.c
)

target_include_directories(test_conv_thunks_lartg_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lartg_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lartg_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lartg_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lartg_family COMMAND test_conv_thunks_lartg_family)

message(STATUS "LAPACK LARTG convention thunk test enabled")

# ============================================================================
# LAPACK LARTV Convention Thunks — focused synthetic coverage for
# real and complex strided vector-rotation helpers without info returns.
# ============================================================================
add_executable(test_conv_thunks_lartv_family
    test_conv_thunks_lartv_family.c
)

target_include_directories(test_conv_thunks_lartv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lartv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lartv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lartv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lartv_family COMMAND test_conv_thunks_lartv_family)

message(STATUS "LAPACK LARTV convention thunk test enabled")

# ============================================================================
# LAPACK LAS2 Convention Thunks — focused synthetic coverage for
# small singular-value helpers with scalar-by-value inputs and no info return.
# ============================================================================
add_executable(test_conv_thunks_las2_family
    test_conv_thunks_las2_family.c
)

target_include_directories(test_conv_thunks_las2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_las2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_las2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_las2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_las2_family COMMAND test_conv_thunks_las2_family)

message(STATUS "LAPACK LAS2 convention thunk test enabled")

# ============================================================================
# LAPACK LASCL Convention Thunks — focused synthetic coverage for
# real and complex matrix scaling helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lascl_family
    test_conv_thunks_lascl_family.c
)

target_include_directories(test_conv_thunks_lascl_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lascl_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lascl_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lascl_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lascl_family COMMAND test_conv_thunks_lascl_family)

message(STATUS "LAPACK LASCL convention thunk test enabled")

# ============================================================================
# LAPACK LASD0 Convention Thunks — focused synthetic coverage for
# real divide-and-conquer setup helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd0_family
    test_conv_thunks_lasd0_family.c
)

target_include_directories(test_conv_thunks_lasd0_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd0_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd0_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd0_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd0_family COMMAND test_conv_thunks_lasd0_family)

message(STATUS "LAPACK LASD0 convention thunk test enabled")

# ============================================================================
# LAPACK LASD1 Convention Thunks — focused synthetic coverage for
# real divide-and-conquer merge helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd1_family
    test_conv_thunks_lasd1_family.c
)

target_include_directories(test_conv_thunks_lasd1_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd1_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd1_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd1_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd1_family COMMAND test_conv_thunks_lasd1_family)

message(STATUS "LAPACK LASD1 convention thunk test enabled")

# ============================================================================
# LAPACK LASD2 Convention Thunks — focused synthetic coverage for
# real divide-and-conquer deflation helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd2_family
    test_conv_thunks_lasd2_family.c
)

target_include_directories(test_conv_thunks_lasd2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd2_family COMMAND test_conv_thunks_lasd2_family)

message(STATUS "LAPACK LASD2 convention thunk test enabled")

# ============================================================================
# LAPACK LASD3 Convention Thunks — focused synthetic coverage for
# real secular-solve/vector-update helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd3_family
    test_conv_thunks_lasd3_family.c
)

target_include_directories(test_conv_thunks_lasd3_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd3_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd3_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd3_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd3_family COMMAND test_conv_thunks_lasd3_family)

message(STATUS "LAPACK LASD3 convention thunk test enabled")

# ============================================================================
# LAPACK LASD4 Convention Thunks — focused synthetic coverage for
# real secular-equation helpers with scalar rho and info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd4_family
    test_conv_thunks_lasd4_family.c
)

target_include_directories(test_conv_thunks_lasd4_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd4_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd4_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd4_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd4_family COMMAND test_conv_thunks_lasd4_family)

message(STATUS "LAPACK LASD4 convention thunk test enabled")

# ============================================================================
# LAPACK LASD5 Convention Thunks — focused synthetic coverage for
# real 2x2 secular helpers with scalar rho and no info return.
# ============================================================================
add_executable(test_conv_thunks_lasd5_family
    test_conv_thunks_lasd5_family.c
)

target_include_directories(test_conv_thunks_lasd5_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd5_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd5_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd5_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd5_family COMMAND test_conv_thunks_lasd5_family)

message(STATUS "LAPACK LASD5 convention thunk test enabled")

# ============================================================================
# LAPACK LASD6 Convention Thunks — focused synthetic coverage for
# real divide-and-conquer merge drivers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd6_family
    test_conv_thunks_lasd6_family.c
)

target_include_directories(test_conv_thunks_lasd6_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd6_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd6_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd6_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd6_family COMMAND test_conv_thunks_lasd6_family)

message(STATUS "LAPACK LASD6 convention thunk test enabled")

# ============================================================================
# LAPACK LASD7 Convention Thunks — focused synthetic coverage for
# real deflation helpers with input alpha/beta and info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd7_family
    test_conv_thunks_lasd7_family.c
)

target_include_directories(test_conv_thunks_lasd7_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd7_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd7_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd7_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd7_family COMMAND test_conv_thunks_lasd7_family)

message(STATUS "LAPACK LASD7 convention thunk test enabled")

# ============================================================================
# LAPACK LASD8 Convention Thunks — focused synthetic coverage for
# real singular-vector update helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasd8_family
    test_conv_thunks_lasd8_family.c
)

target_include_directories(test_conv_thunks_lasd8_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasd8_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasd8_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasd8_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasd8_family COMMAND test_conv_thunks_lasd8_family)

message(STATUS "LAPACK LASD8 convention thunk test enabled")

# ============================================================================
# LAPACK LASDA Convention Thunks — focused synthetic coverage for
# real divide-and-conquer tree setup helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasda_family
    test_conv_thunks_lasda_family.c
)

target_include_directories(test_conv_thunks_lasda_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasda_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasda_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasda_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasda_family COMMAND test_conv_thunks_lasda_family)

message(STATUS "LAPACK LASDA convention thunk test enabled")

# ============================================================================
# LAPACK LASDQ Convention Thunks — focused synthetic coverage for
# real bidiagonal SVD helpers with char-plus-shape info bridging.
# ============================================================================
add_executable(test_conv_thunks_lasdq_family
    test_conv_thunks_lasdq_family.c
)

target_include_directories(test_conv_thunks_lasdq_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasdq_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasdq_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasdq_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasdq_family COMMAND test_conv_thunks_lasdq_family)

message(STATUS "LAPACK LASDQ convention thunk test enabled")

# ============================================================================
# LAPACK LASDT Convention Thunks — focused synthetic coverage for
# real divide-and-conquer tree-shape helpers without info returns.
# ============================================================================
add_executable(test_conv_thunks_lasdt_family
    test_conv_thunks_lasdt_family.c
)

target_include_directories(test_conv_thunks_lasdt_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasdt_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasdt_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasdt_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasdt_family COMMAND test_conv_thunks_lasdt_family)

message(STATUS "LAPACK LASDT convention thunk test enabled")

# ============================================================================
# LAPACK LASET Convention Thunks — focused synthetic coverage for
# real and complex matrix-initialization helpers without info returns.
# ============================================================================
add_executable(test_conv_thunks_laset_family
    test_conv_thunks_laset_family.c
)

target_include_directories(test_conv_thunks_laset_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laset_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laset_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laset_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laset_family COMMAND test_conv_thunks_laset_family)

message(STATUS "LAPACK LASET convention thunk test enabled")

# ============================================================================
# LAPACK LASQ1 Convention Thunks — focused synthetic coverage for
# real singular-value preprocessing helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasq1_family
    test_conv_thunks_lasq1_family.c
)

target_include_directories(test_conv_thunks_lasq1_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasq1_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasq1_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasq1_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasq1_family COMMAND test_conv_thunks_lasq1_family)

message(STATUS "LAPACK LASQ1 convention thunk test enabled")

# ============================================================================
# LAPACK LASQ2 Convention Thunks — focused synthetic coverage for
# real qd-array preprocessing helpers with info-return bridging.
# ============================================================================
add_executable(test_conv_thunks_lasq2_family
    test_conv_thunks_lasq2_family.c
)

target_include_directories(test_conv_thunks_lasq2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasq2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasq2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasq2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasq2_family COMMAND test_conv_thunks_lasq2_family)

message(STATUS "LAPACK LASQ2 convention thunk test enabled")

# ============================================================================
# LAPACK LASQ3 Convention Thunks — focused synthetic coverage for
# real dqds iteration helpers with mixed input and in-out scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_lasq3_family
    test_conv_thunks_lasq3_family.c
)

target_include_directories(test_conv_thunks_lasq3_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasq3_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasq3_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasq3_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasq3_family COMMAND test_conv_thunks_lasq3_family)

message(STATUS "LAPACK LASQ3 convention thunk test enabled")

# ============================================================================
# LAPACK LASQ4 Convention Thunks — focused synthetic coverage for
# real dqds shift-selection helpers with scalar input normalization.
# ============================================================================
add_executable(test_conv_thunks_lasq4_family
    test_conv_thunks_lasq4_family.c
)

target_include_directories(test_conv_thunks_lasq4_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasq4_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasq4_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasq4_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasq4_family COMMAND test_conv_thunks_lasq4_family)

message(STATUS "LAPACK LASQ4 convention thunk test enabled")

# ============================================================================
# LAPACK LASQ5 Convention Thunks — focused synthetic coverage for
# real dqds sweep helpers with in-out qd arrays and minima state.
# ============================================================================
add_executable(test_conv_thunks_lasq5_family
    test_conv_thunks_lasq5_family.c
)

target_include_directories(test_conv_thunks_lasq5_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasq5_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasq5_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasq5_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasq5_family COMMAND test_conv_thunks_lasq5_family)

message(STATUS "LAPACK LASQ5 convention thunk test enabled")

# ============================================================================
# LAPACK LASQ6 Convention Thunks — focused synthetic coverage for
# real dqds cleanup helpers with int-return bridging on both sides.
# ============================================================================
add_executable(test_conv_thunks_lasq6_family
    test_conv_thunks_lasq6_family.c
)

target_include_directories(test_conv_thunks_lasq6_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasq6_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasq6_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasq6_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasq6_family COMMAND test_conv_thunks_lasq6_family)

message(STATUS "LAPACK LASQ6 convention thunk test enabled")

# ============================================================================
# LAPACK LASR Convention Thunks — focused synthetic coverage for
# real plane-rotation matrix helpers with selector chars by value.
# ============================================================================
add_executable(test_conv_thunks_lasr_family
    test_conv_thunks_lasr_family.c
)

target_include_directories(test_conv_thunks_lasr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasr_family COMMAND test_conv_thunks_lasr_family)

message(STATUS "LAPACK LASR convention thunk test enabled")

# ============================================================================
# LAPACK LAEXC Convention Thunks — focused synthetic coverage for
# no-layout block-exchange helpers that return info through the C int return.
# ============================================================================
add_executable(test_conv_thunks_laexc_family
    test_conv_thunks_laexc_family.c
)

target_include_directories(test_conv_thunks_laexc_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laexc_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laexc_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laexc_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laexc_family COMMAND test_conv_thunks_laexc_family)

message(STATUS "LAPACK LAEXC convention thunk test enabled")

# ============================================================================
# LAPACK LAESY Convention Thunks — focused synthetic coverage for
# complex scalar eigensystem bridges with no layout translation or info slot.
# ============================================================================
add_executable(test_conv_thunks_laesy_family
    test_conv_thunks_laesy_family.c
)

target_include_directories(test_conv_thunks_laesy_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laesy_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laesy_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laesy_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laesy_family COMMAND test_conv_thunks_laesy_family)

message(STATUS "LAPACK LAESY convention thunk test enabled")

# ============================================================================
# LAPACK LAIC1 Convention Thunks — focused synthetic coverage for
# vector-plus-scalar incremental-condition bridges with no layout translation.
# ============================================================================
add_executable(test_conv_thunks_laic1_family
    test_conv_thunks_laic1_family.c
)

target_include_directories(test_conv_thunks_laic1_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laic1_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laic1_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laic1_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laic1_family COMMAND test_conv_thunks_laic1_family)

message(STATUS "LAPACK LAIC1 convention thunk test enabled")

# ============================================================================
# LAPACK LALN2 Convention Thunks — focused synthetic coverage for
# no-layout small-system helpers with info returned through the C int return.
# ============================================================================
add_executable(test_conv_thunks_laln2_family
    test_conv_thunks_laln2_family.c
)

target_include_directories(test_conv_thunks_laln2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laln2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laln2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laln2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laln2_family COMMAND test_conv_thunks_laln2_family)

message(STATUS "LAPACK LALN2 convention thunk test enabled")

# ============================================================================
# LAPACK LACPY Convention Thunks — focused synthetic coverage for
# matrix-copy bridges with row-major transpose staging.
# ============================================================================
add_executable(test_conv_thunks_lacpy_family
    test_conv_thunks_lacpy_family.c
)

target_include_directories(test_conv_thunks_lacpy_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lacpy_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lacpy_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lacpy_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lacpy_family COMMAND test_conv_thunks_lacpy_family)

message(STATUS "LAPACK LACPY convention thunk test enabled")

# ============================================================================
# LAPACK LASRT Convention Thunks — focused synthetic coverage for
# char-plus-vector sort bridges with int/info propagation.
# ============================================================================
add_executable(test_conv_thunks_lasrt_family
    test_conv_thunks_lasrt_family.c
)

target_include_directories(test_conv_thunks_lasrt_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lasrt_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lasrt_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lasrt_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lasrt_family COMMAND test_conv_thunks_lasrt_family)

message(STATUS "LAPACK LASRT convention thunk test enabled")

# ============================================================================
# LAPACK LASWP Convention Thunks — focused synthetic coverage for
# row-swap bridges with row-major matrix translation and copy-back.
# ============================================================================
add_executable(test_conv_thunks_laswp_family
    test_conv_thunks_laswp_family.c
)

target_include_directories(test_conv_thunks_laswp_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_laswp_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_laswp_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_laswp_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_laswp_family COMMAND test_conv_thunks_laswp_family)

message(STATUS "LAPACK LASWP convention thunk test enabled")

# ============================================================================
# Bidiagonal Reduction Convention Thunks — focused synthetic coverage for
# GEBRD bridges with D/E/TAUQ/TAUP outputs and workspace-query handling.
# ============================================================================
add_executable(test_conv_thunks_bidiag_reduction_family
    test_conv_thunks_bidiag_reduction_family.c
)

target_include_directories(test_conv_thunks_bidiag_reduction_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_bidiag_reduction_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_bidiag_reduction_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_bidiag_reduction_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_bidiag_reduction_family COMMAND test_conv_thunks_bidiag_reduction_family)

message(STATUS "Bidiagonal reduction convention thunk test enabled")

# ============================================================================
# Condition-Estimation Convention Thunks — focused synthetic coverage for
# GECON bridges with internal workspace allocation and reverse all-pointer ABI.
# ============================================================================
add_executable(test_conv_thunks_gecon_family
    test_conv_thunks_gecon_family.c
)

target_include_directories(test_conv_thunks_gecon_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gecon_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gecon_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gecon_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gecon_family COMMAND test_conv_thunks_gecon_family)

message(STATUS "Condition-estimation convention thunk test enabled")

# ============================================================================
# Matrix Equilibration Convention Thunks — focused synthetic coverage for
# GEEQU bridges across real and complex variants.
# ============================================================================
add_executable(test_conv_thunks_geequ_family
    test_conv_thunks_geequ_family.c
)

target_include_directories(test_conv_thunks_geequ_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_geequ_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_geequ_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_geequ_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_geequ_family COMMAND test_conv_thunks_geequ_family)

message(STATUS "Matrix equilibration convention thunk test enabled")

# ============================================================================
# Eigenvalue Convention Thunks — focused synthetic coverage for GEEV bridges
# with workspace queries and reverse all-pointer ABI.
# ============================================================================
add_executable(test_conv_thunks_geev_family
    test_conv_thunks_geev_family.c
)

target_include_directories(test_conv_thunks_geev_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_geev_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_geev_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_geev_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_geev_family COMMAND test_conv_thunks_geev_family)

message(STATUS "Eigenvalue convention thunk test enabled")

# ============================================================================
# Singular-Value Convention Thunks — focused synthetic coverage for GESVD
# bridges, including superb propagation and reverse scratch allocation.
# ============================================================================
add_executable(test_conv_thunks_gesvd_family
    test_conv_thunks_gesvd_family.c
)

target_include_directories(test_conv_thunks_gesvd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gesvd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gesvd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gesvd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gesvd_family COMMAND test_conv_thunks_gesvd_family)

message(STATUS "Singular-value convention thunk test enabled")

# ============================================================================
# Divide-and-Conquer SVD Convention Thunks — focused synthetic coverage for
# GESDD bridges with workspace queries and reverse all-pointer ABI.
# ============================================================================
add_executable(test_conv_thunks_gesdd_family
    test_conv_thunks_gesdd_family.c
)

target_include_directories(test_conv_thunks_gesdd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gesdd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gesdd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gesdd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gesdd_family COMMAND test_conv_thunks_gesdd_family)

message(STATUS "Divide-and-conquer SVD convention thunk test enabled")

# ============================================================================
# Symmetric/Hermitian Eigenvalue Convention Thunks — focused synthetic
# coverage for SYEV/HEEV bridges with workspace queries and reverse uplo mapping.
# ============================================================================
add_executable(test_conv_thunks_syev_family
    test_conv_thunks_syev_family.c
)

target_include_directories(test_conv_thunks_syev_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_syev_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_syev_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_syev_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_syev_family COMMAND test_conv_thunks_syev_family)

message(STATUS "Symmetric/Hermitian eigenvalue convention thunk test enabled")

# ============================================================================
# Packed Symmetric Divide-and-Conquer Eigen Convention Thunks — focused
# coverage for SSPEVD/DSPEVD with packed row-major reorder, queried WORK/IWORK,
# and optional row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_spevd_family
    test_conv_thunks_spevd_family.c
)

target_include_directories(test_conv_thunks_spevd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spevd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spevd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spevd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spevd_family COMMAND test_conv_thunks_spevd_family)

message(STATUS "Packed symmetric divide-and-conquer convention thunk test enabled")

# ============================================================================
# Packed Symmetric Generalized Selection Eigen Convention Thunks — focused
# coverage for SSPGVX/DSPGVX with packed row-major A/B reorder, hidden
# WORK/IWORK allocation, and range-dependent eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_spgvx_family
    test_conv_thunks_spgvx_family.c
)

target_include_directories(test_conv_thunks_spgvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spgvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spgvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spgvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spgvx_family COMMAND test_conv_thunks_spgvx_family)

message(STATUS "Packed symmetric generalized selection convention thunk test enabled")

# ============================================================================
# Packed Hermitian Eigen Convention Thunks — focused coverage for CHPEV/ZHPEV
# with packed row-major reorder, hidden WORK/RWORK allocation, and eigenvector
# copy-back.
# ============================================================================
add_executable(test_conv_thunks_hpev_family
    test_conv_thunks_hpev_family.c
)

target_include_directories(test_conv_thunks_hpev_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpev_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpev_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpev_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpev_family COMMAND test_conv_thunks_hpev_family)

message(STATUS "Packed Hermitian eigen convention thunk test enabled")

# ============================================================================
# Packed Hermitian Divide-and-Conquer Eigen Convention Thunks — focused
# coverage for CHPEVD/ZHPEVD with packed row-major reorder, query-based
# WORK/RWORK/IWORK allocation, and eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hpevd_family
    test_conv_thunks_hpevd_family.c
)

target_include_directories(test_conv_thunks_hpevd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpevd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpevd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpevd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpevd_family COMMAND test_conv_thunks_hpevd_family)

message(STATUS "Packed Hermitian divide-and-conquer convention thunk test enabled")

# ============================================================================
# Packed Hermitian Selection Eigen Convention Thunks — focused coverage for
# CHPEVX/ZHPEVX with packed row-major reorder, hidden WORK/RWORK/IWORK
# allocation, and range-dependent eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hpevx_family
    test_conv_thunks_hpevx_family.c
)

target_include_directories(test_conv_thunks_hpevx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpevx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpevx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpevx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpevx_family COMMAND test_conv_thunks_hpevx_family)

message(STATUS "Packed Hermitian selection convention thunk test enabled")

# ============================================================================
# Packed Hermitian Generalized Eigen Convention Thunks — focused coverage for
# CHPGV/ZHPGV with packed row-major reorder for A/B, hidden WORK/RWORK
# allocation, and eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hpgv_family
    test_conv_thunks_hpgv_family.c
)

target_include_directories(test_conv_thunks_hpgv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpgv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpgv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpgv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpgv_family COMMAND test_conv_thunks_hpgv_family)

message(STATUS "Packed Hermitian generalized convention thunk test enabled")

# ============================================================================
# Packed Hermitian Generalized Divide-and-Conquer Eigen Convention Thunks —
# focused coverage for CHPGVD/ZHPGVD with packed row-major reorder for A/B,
# query-based WORK/RWORK/IWORK allocation, and eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hpgvd_family
    test_conv_thunks_hpgvd_family.c
)

target_include_directories(test_conv_thunks_hpgvd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpgvd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpgvd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpgvd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpgvd_family COMMAND test_conv_thunks_hpgvd_family)

message(STATUS "Packed Hermitian generalized divide-and-conquer convention thunk test enabled")

# ============================================================================
# Packed Hermitian Generalized Selection Eigen Convention Thunks — focused
# coverage for CHPGVX/ZHPGVX with packed row-major reorder for A/B, hidden
# WORK/RWORK/IWORK allocation, and range-dependent eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hpgvx_family
    test_conv_thunks_hpgvx_family.c
)

target_include_directories(test_conv_thunks_hpgvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpgvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpgvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpgvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpgvx_family COMMAND test_conv_thunks_hpgvx_family)

message(STATUS "Packed Hermitian generalized selection convention thunk test enabled")

# ============================================================================
# Packed Symmetric Generalized Eigen Convention Thunks — focused coverage for
# SSPGV/DSPGV with packed row-major reorder for A/B, hidden WORK allocation,
# and optional row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_spgv_family
    test_conv_thunks_spgv_family.c
)

target_include_directories(test_conv_thunks_spgv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spgv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spgv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spgv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spgv_family COMMAND test_conv_thunks_spgv_family)

message(STATUS "Packed symmetric generalized eigen convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal EVR convention thunks — focused synthetic coverage for
# SSTEVR/DSTEVR query and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_stevr_family
    test_conv_thunks_stevr_family.c
)

target_include_directories(test_conv_thunks_stevr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stevr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stevr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stevr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stevr_family COMMAND test_conv_thunks_stevr_family)

message(STATUS "Symmetric tridiagonal EVR convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal EVX convention thunks — focused synthetic coverage for
# SSTEVX/DSTEVX fixed-workspace and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_stevx_family
    test_conv_thunks_stevx_family.c
)

target_include_directories(test_conv_thunks_stevx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stevx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stevx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stevx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stevx_family COMMAND test_conv_thunks_stevx_family)

message(STATUS "Symmetric tridiagonal EVX convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal STEBZ convention thunks — focused synthetic coverage
# for SSTEBZ/DSTEBZ fixed-workspace and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_stebz_family
    test_conv_thunks_stebz_family.c
)

target_include_directories(test_conv_thunks_stebz_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stebz_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stebz_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stebz_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stebz_family COMMAND test_conv_thunks_stebz_family)

message(STATUS "Symmetric tridiagonal STEBZ convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal STEIN convention thunks — focused synthetic coverage
# for SSTEIN/DSTEIN fixed-workspace and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_stein_family
    test_conv_thunks_stein_family.c
)

target_include_directories(test_conv_thunks_stein_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stein_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stein_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stein_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stein_family COMMAND test_conv_thunks_stein_family)

message(STATUS "Symmetric tridiagonal STEIN convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal STERF convention thunks — focused synthetic coverage
# for SSTERF/DSTERF direct eigenvalue bridges.
# ============================================================================
add_executable(test_conv_thunks_sterf_family
    test_conv_thunks_sterf_family.c
)

target_include_directories(test_conv_thunks_sterf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sterf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sterf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sterf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sterf_family COMMAND test_conv_thunks_sterf_family)

message(STATUS "Symmetric tridiagonal STERF convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal STEQR convention thunks — focused synthetic coverage
# for SSTEQR/DSTEQR bridges with fixed workspace and row-major vector handling.
# ============================================================================
add_executable(test_conv_thunks_steqr_family
    test_conv_thunks_steqr_family.c
)

target_include_directories(test_conv_thunks_steqr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_steqr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_steqr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_steqr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_steqr_family COMMAND test_conv_thunks_steqr_family)

message(STATUS "Symmetric tridiagonal STEQR convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal STEGR convention thunks — focused synthetic coverage
# for SSTEGR/DSTEGR query bridging and row-major vector export.
# ============================================================================
add_executable(test_conv_thunks_stegr_family
    test_conv_thunks_stegr_family.c
)

target_include_directories(test_conv_thunks_stegr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stegr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stegr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stegr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stegr_family COMMAND test_conv_thunks_stegr_family)

message(STATUS "Symmetric tridiagonal STEGR convention thunk test enabled")

# ============================================================================
# Symmetric tridiagonal STEDC convention thunks — focused synthetic coverage
# for SSTEDC/DSTEDC query bridging and row-major vector handling.
# ============================================================================
add_executable(test_conv_thunks_stedc_family
    test_conv_thunks_stedc_family.c
)

target_include_directories(test_conv_thunks_stedc_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stedc_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stedc_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stedc_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stedc_family COMMAND test_conv_thunks_stedc_family)

message(STATUS "Symmetric tridiagonal STEDC convention thunk test enabled")

# ============================================================================
# Complex symmetric tridiagonal STEQR convention thunks — focused synthetic
# coverage for CSTEQR/ZSTEQR fixed-work bridging and row-major vector handling.
# ============================================================================
add_executable(test_conv_thunks_steqr_complex_family
    test_conv_thunks_steqr_complex_family.c
)

target_include_directories(test_conv_thunks_steqr_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_steqr_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_steqr_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_steqr_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_steqr_complex_family COMMAND test_conv_thunks_steqr_complex_family)

message(STATUS "Complex symmetric tridiagonal STEQR convention thunk test enabled")

# ============================================================================
# Complex symmetric tridiagonal STEDC convention thunks — focused synthetic
# coverage for CSTEDC/ZSTEDC query bridging and row-major vector handling.
# ============================================================================
add_executable(test_conv_thunks_stedc_complex_family
    test_conv_thunks_stedc_complex_family.c
)

target_include_directories(test_conv_thunks_stedc_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stedc_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stedc_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stedc_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stedc_complex_family COMMAND test_conv_thunks_stedc_complex_family)

message(STATUS "Complex symmetric tridiagonal STEDC convention thunk test enabled")

# ============================================================================
# Complex symmetric tridiagonal STEGR convention thunks — focused synthetic
# coverage for CSTEGR/ZSTEGR query bridging and row-major vector export.
# ============================================================================
add_executable(test_conv_thunks_stegr_complex_family
    test_conv_thunks_stegr_complex_family.c
)

target_include_directories(test_conv_thunks_stegr_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stegr_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stegr_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stegr_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stegr_complex_family COMMAND test_conv_thunks_stegr_complex_family)

message(STATUS "Complex symmetric tridiagonal STEGR convention thunk test enabled")

# ============================================================================
# Complex symmetric tridiagonal STEIN convention thunks — focused synthetic
# coverage for CSTEIN/ZSTEIN fixed-work bridging and row-major vector export.
# ============================================================================
add_executable(test_conv_thunks_stein_complex_family
    test_conv_thunks_stein_complex_family.c
)

target_include_directories(test_conv_thunks_stein_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_stein_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_stein_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_stein_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_stein_complex_family COMMAND test_conv_thunks_stein_complex_family)

message(STATUS "Complex symmetric tridiagonal STEIN convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SYR convention thunks — focused synthetic coverage
# for CSYR/ZSYR pointer-based scalar and buffer bridging.
# ============================================================================
add_executable(test_conv_thunks_syr_complex_family
    test_conv_thunks_syr_complex_family.c
)

target_include_directories(test_conv_thunks_syr_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_syr_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_syr_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_syr_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_syr_complex_family COMMAND test_conv_thunks_syr_complex_family)

message(STATUS "Complex symmetric BLAS SYR convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SYR2 convention thunks — focused synthetic coverage
# for CSYR2/ZSYR2 pointer-based scalar and buffer bridging.
# ============================================================================
add_executable(test_conv_thunks_syr2_complex_family
    test_conv_thunks_syr2_complex_family.c
)

target_include_directories(test_conv_thunks_syr2_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_syr2_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_syr2_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_syr2_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_syr2_complex_family COMMAND test_conv_thunks_syr2_complex_family)

message(STATUS "Complex symmetric BLAS SYR2 convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HEMV convention thunks — focused synthetic coverage
# for CHEMV/ZHEMV by-value scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_hemv_complex_family
    test_conv_thunks_hemv_complex_family.c
)

target_include_directories(test_conv_thunks_hemv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hemv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hemv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hemv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hemv_complex_family COMMAND test_conv_thunks_hemv_complex_family)

message(STATUS "Complex Hermitian BLAS HEMV convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HPMV convention thunks — focused synthetic coverage
# for CHPMV/ZHPMV packed-storage by-value scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_hpmv_complex_family
    test_conv_thunks_hpmv_complex_family.c
)

target_include_directories(test_conv_thunks_hpmv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpmv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpmv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpmv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpmv_complex_family COMMAND test_conv_thunks_hpmv_complex_family)

message(STATUS "Complex Hermitian BLAS HPMV convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HBMV convention thunks — focused synthetic coverage
# for CHBMV/ZHBMV Hermitian-band by-value scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_hbmv_complex_family
    test_conv_thunks_hbmv_complex_family.c
)

target_include_directories(test_conv_thunks_hbmv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hbmv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hbmv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hbmv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hbmv_complex_family COMMAND test_conv_thunks_hbmv_complex_family)

message(STATUS "Complex Hermitian BLAS HBMV convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SBMV convention thunks — focused synthetic coverage
# for CSBMV/ZSBMV symmetric-band by-value scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_sbmv_complex_family
    test_conv_thunks_sbmv_complex_family.c
)

target_include_directories(test_conv_thunks_sbmv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbmv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbmv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbmv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbmv_complex_family COMMAND test_conv_thunks_sbmv_complex_family)

message(STATUS "Complex symmetric BLAS SBMV convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HPR convention thunks — focused synthetic coverage
# for CHPR/ZHPR packed rank-1 updates with real alpha bridging.
# ============================================================================
add_executable(test_conv_thunks_hpr_complex_family
    test_conv_thunks_hpr_complex_family.c
)

target_include_directories(test_conv_thunks_hpr_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpr_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpr_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpr_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpr_complex_family COMMAND test_conv_thunks_hpr_complex_family)

message(STATUS "Complex Hermitian BLAS HPR convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HPR2 convention thunks — focused synthetic coverage
# for CHPR2/ZHPR2 packed rank-2 updates with complex alpha bridging.
# ============================================================================
add_executable(test_conv_thunks_hpr2_complex_family
    test_conv_thunks_hpr2_complex_family.c
)

target_include_directories(test_conv_thunks_hpr2_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hpr2_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hpr2_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hpr2_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hpr2_complex_family COMMAND test_conv_thunks_hpr2_complex_family)

message(STATUS "Complex Hermitian BLAS HPR2 convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HER convention thunks — focused synthetic coverage
# for CHER/ZHER full-matrix rank-1 updates with real alpha bridging.
# ============================================================================
add_executable(test_conv_thunks_her_complex_family
    test_conv_thunks_her_complex_family.c
)

target_include_directories(test_conv_thunks_her_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_her_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_her_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_her_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_her_complex_family COMMAND test_conv_thunks_her_complex_family)

message(STATUS "Complex Hermitian BLAS HER convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HER2 convention thunks — focused synthetic coverage
# for CHER2/ZHER2 full-matrix rank-2 updates with complex alpha bridging.
# ============================================================================
add_executable(test_conv_thunks_her2_complex_family
    test_conv_thunks_her2_complex_family.c
)

target_include_directories(test_conv_thunks_her2_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_her2_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_her2_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_her2_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_her2_complex_family COMMAND test_conv_thunks_her2_complex_family)

message(STATUS "Complex Hermitian BLAS HER2 convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HERK convention thunks — focused synthetic coverage
# for CHERK/ZHERK rank-k updates with real alpha/beta bridging.
# ============================================================================
add_executable(test_conv_thunks_herk_complex_family
    test_conv_thunks_herk_complex_family.c
)

target_include_directories(test_conv_thunks_herk_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_herk_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_herk_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_herk_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_herk_complex_family COMMAND test_conv_thunks_herk_complex_family)

message(STATUS "Complex Hermitian BLAS HERK convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HER2K convention thunks — focused synthetic coverage
# for CHER2K/ZHER2K rank-2k updates with complex alpha and real beta bridging.
# ============================================================================
add_executable(test_conv_thunks_her2k_complex_family
    test_conv_thunks_her2k_complex_family.c
)

target_include_directories(test_conv_thunks_her2k_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_her2k_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_her2k_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_her2k_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_her2k_complex_family COMMAND test_conv_thunks_her2k_complex_family)

message(STATUS "Complex Hermitian BLAS HER2K convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SYMM convention thunks — focused synthetic coverage
# for CSYMM/ZSYMM matrix-matrix bridging with by-value complex alpha/beta.
# ============================================================================
add_executable(test_conv_thunks_symm_complex_family
    test_conv_thunks_symm_complex_family.c
)

target_include_directories(test_conv_thunks_symm_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_symm_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_symm_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_symm_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_symm_complex_family COMMAND test_conv_thunks_symm_complex_family)

message(STATUS "Complex symmetric BLAS SYMM convention thunk test enabled")

# ============================================================================
# Complex Hermitian BLAS HEMM convention thunks — focused synthetic coverage
# for CHEMM/ZHEMM matrix-matrix bridging with pointer-form complex alpha/beta.
# ============================================================================
add_executable(test_conv_thunks_hemm_complex_family
    test_conv_thunks_hemm_complex_family.c
)

target_include_directories(test_conv_thunks_hemm_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hemm_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hemm_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hemm_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hemm_complex_family COMMAND test_conv_thunks_hemm_complex_family)

message(STATUS "Complex Hermitian BLAS HEMM convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SYRK convention thunks — focused synthetic coverage
# for CSYRK/ZSYRK rank-k bridging with pointer-form complex alpha/beta.
# ============================================================================
add_executable(test_conv_thunks_syrk_complex_family
    test_conv_thunks_syrk_complex_family.c
)

target_include_directories(test_conv_thunks_syrk_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_syrk_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_syrk_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_syrk_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_syrk_complex_family COMMAND test_conv_thunks_syrk_complex_family)

message(STATUS "Complex symmetric BLAS SYRK convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SYR2K convention thunks — focused synthetic coverage
# for CSYR2K/ZSYR2K rank-2k bridging with pointer-form complex alpha/beta.
# ============================================================================
add_executable(test_conv_thunks_syr2k_complex_family
    test_conv_thunks_syr2k_complex_family.c
)

target_include_directories(test_conv_thunks_syr2k_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_syr2k_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_syr2k_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_syr2k_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_syr2k_complex_family COMMAND test_conv_thunks_syr2k_complex_family)

message(STATUS "Complex symmetric BLAS SYR2K convention thunk test enabled")

# ============================================================================
# Complex triangular BLAS TRMM convention thunks — focused synthetic coverage
# for CTRMM/ZTRMM bridging with pointer-form complex alpha.
# ============================================================================
add_executable(test_conv_thunks_trmm_complex_family
    test_conv_thunks_trmm_complex_family.c
)

target_include_directories(test_conv_thunks_trmm_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_trmm_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_trmm_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_trmm_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_trmm_complex_family COMMAND test_conv_thunks_trmm_complex_family)

message(STATUS "Complex triangular BLAS TRMM convention thunk test enabled")

# ============================================================================
# Complex triangular BLAS TRSM convention thunks — focused synthetic coverage
# for CTRSM/ZTRSM bridging with pointer-form complex alpha.
# ============================================================================
add_executable(test_conv_thunks_trsm_complex_family
    test_conv_thunks_trsm_complex_family.c
)

target_include_directories(test_conv_thunks_trsm_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_trsm_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_trsm_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_trsm_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_trsm_complex_family COMMAND test_conv_thunks_trsm_complex_family)

message(STATUS "Complex triangular BLAS TRSM convention thunk test enabled")

# ============================================================================
# Complex triangular BLAS TRMV/TRSV convention thunks — focused synthetic
# coverage for CTRMV/ZTRMV/CTRSV/ZTRSV bridge installation and forwarding.
# ============================================================================
add_executable(test_conv_thunks_trmv_trsv_complex_family
    test_conv_thunks_trmv_trsv_complex_family.c
)

target_include_directories(test_conv_thunks_trmv_trsv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_trmv_trsv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_trmv_trsv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_trmv_trsv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_trmv_trsv_complex_family COMMAND test_conv_thunks_trmv_trsv_complex_family)

message(STATUS "Complex triangular BLAS TRMV/TRSV convention thunk test enabled")

# ============================================================================
# Complex triangular band BLAS TBMV/TBSV convention thunks — focused synthetic
# coverage for CTBMV/ZTBMV/CTBSV/ZTBSV bridge installation and forwarding.
# ============================================================================
add_executable(test_conv_thunks_tbmv_tbsv_complex_family
    test_conv_thunks_tbmv_tbsv_complex_family.c
)

target_include_directories(test_conv_thunks_tbmv_tbsv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_tbmv_tbsv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_tbmv_tbsv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_tbmv_tbsv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_tbmv_tbsv_complex_family COMMAND test_conv_thunks_tbmv_tbsv_complex_family)

message(STATUS "Complex triangular band BLAS TBMV/TBSV convention thunk test enabled")

# ============================================================================
# Complex triangular packed BLAS TPMV/TPSV convention thunks — focused
# synthetic coverage for CTPMV/ZTPMV/CTPSV/ZTPSV bridge installation.
# ============================================================================
add_executable(test_conv_thunks_tpmv_tpsv_complex_family
    test_conv_thunks_tpmv_tpsv_complex_family.c
)

target_include_directories(test_conv_thunks_tpmv_tpsv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_tpmv_tpsv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_tpmv_tpsv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_tpmv_tpsv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_tpmv_tpsv_complex_family COMMAND test_conv_thunks_tpmv_tpsv_complex_family)

message(STATUS "Complex triangular packed BLAS TPMV/TPSV convention thunk test enabled")

# ============================================================================
# Complex general BLAS GEMV/GER convention thunks — focused synthetic coverage
# for CGEMV/ZGEMV/CGERC/ZGERC/CGERU/ZGERU pointer-scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_gemv_ger_complex_family
    test_conv_thunks_gemv_ger_complex_family.c
)

target_include_directories(test_conv_thunks_gemv_ger_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gemv_ger_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gemv_ger_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gemv_ger_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gemv_ger_complex_family COMMAND test_conv_thunks_gemv_ger_complex_family)

message(STATUS "Complex general BLAS GEMV/GER convention thunk test enabled")

# ============================================================================
# Real symmetric BLAS SBMV/SPMV convention thunks — focused synthetic coverage
# for SSBMV/DSBMV/SSPMV/DSPMV real-scalar bridging.
# ============================================================================
add_executable(test_conv_thunks_sbmv_spmv_real_family
    test_conv_thunks_sbmv_spmv_real_family.c
)

target_include_directories(test_conv_thunks_sbmv_spmv_real_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbmv_spmv_real_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbmv_spmv_real_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbmv_spmv_real_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbmv_spmv_real_family COMMAND test_conv_thunks_sbmv_spmv_real_family)

message(STATUS "Real symmetric BLAS SBMV/SPMV convention thunk test enabled")

# ============================================================================
# Real symmetric packed BLAS SPR/SPR2 convention thunks — focused synthetic
# coverage for SSPR/DSPR/SSPR2/DSPR2 packed update bridging.
# ============================================================================
add_executable(test_conv_thunks_spr_spr2_real_family
    test_conv_thunks_spr_spr2_real_family.c
)

target_include_directories(test_conv_thunks_spr_spr2_real_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spr_spr2_real_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spr_spr2_real_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spr_spr2_real_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spr_spr2_real_family COMMAND test_conv_thunks_spr_spr2_real_family)

message(STATUS "Real symmetric packed BLAS SPR/SPR2 convention thunk test enabled")

# ============================================================================
# BLAS GBMV convention thunks — focused synthetic coverage for
# SGBMV/DGBMV/CGBMV/ZGBMV bridging.
# ============================================================================
add_executable(test_conv_thunks_gbmv_family
    test_conv_thunks_gbmv_family.c
)

target_include_directories(test_conv_thunks_gbmv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbmv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbmv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbmv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbmv_family COMMAND test_conv_thunks_gbmv_family)

message(STATUS "BLAS GBMV convention thunk test enabled")

# ============================================================================
# LAPACK GEBAK/GEBAL convention thunks — focused synthetic coverage for
# S/D/C/Z GEBAK and GEBAL bridging.
# ============================================================================
add_executable(test_conv_thunks_gebal_gebak_family
    test_conv_thunks_gebal_gebak_family.c
)

target_include_directories(test_conv_thunks_gebal_gebak_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gebal_gebak_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gebal_gebak_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gebal_gebak_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gebal_gebak_family COMMAND test_conv_thunks_gebal_gebak_family)

message(STATUS "LAPACK GEBAK/GEBAL convention thunk test enabled")

# ============================================================================
# LAPACK GEBD2 convention thunks — focused synthetic coverage for
# S/D/C/Z GEBD2 bridging.
# ============================================================================
add_executable(test_conv_thunks_gebd2_family
    test_conv_thunks_gebd2_family.c
)

target_include_directories(test_conv_thunks_gebd2_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gebd2_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gebd2_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gebd2_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gebd2_family COMMAND test_conv_thunks_gebd2_family)

message(STATUS "LAPACK GEBD2 convention thunk test enabled")

# ============================================================================
# Complex symmetric BLAS SYMV convention thunks — focused synthetic coverage
# for CSYMV/ZSYMV pointer-based scalar and buffer bridging.
# ============================================================================
add_executable(test_conv_thunks_symv_complex_family
    test_conv_thunks_symv_complex_family.c
)

target_include_directories(test_conv_thunks_symv_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_symv_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_symv_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_symv_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_symv_complex_family COMMAND test_conv_thunks_symv_complex_family)

message(STATUS "Complex symmetric BLAS SYMV convention thunk test enabled")

# ============================================================================
# Symmetric/Hermitian EVD Convention Thunks — focused synthetic coverage for
# SYEVD/HEEVD bridges with workspace queries and reverse uplo mapping.
# ============================================================================
add_executable(test_conv_thunks_syevd_family
    test_conv_thunks_syevd_family.c
)

target_include_directories(test_conv_thunks_syevd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_syevd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_syevd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_syevd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_syevd_family COMMAND test_conv_thunks_syevd_family)

message(STATUS "Symmetric/Hermitian EVD convention thunk test enabled")

# ============================================================================
# Packed Symmetric Selection Eigen Convention Thunks — focused coverage for
# SSPEVX/DSPEVX with packed row-major reorder, hidden WORK/IWORK allocation,
# and range-dependent eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_spevx_family
    test_conv_thunks_spevx_family.c
)

target_include_directories(test_conv_thunks_spevx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spevx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spevx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spevx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spevx_family COMMAND test_conv_thunks_spevx_family)

message(STATUS "Packed symmetric selection convention thunk test enabled")

# ============================================================================
# Packed Symmetric Eigen Convention Thunks — focused coverage for SSPEV/DSPEV
# with packed row-major reorder, hidden WORK allocation, and optional row-major
# eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_spev_family
    test_conv_thunks_spev_family.c
)

target_include_directories(test_conv_thunks_spev_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spev_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spev_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spev_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spev_family COMMAND test_conv_thunks_spev_family)

message(STATUS "Packed symmetric eigen convention thunk test enabled")

# ============================================================================
# Packed Symmetric Generalized Divide-and-Conquer Eigen Convention Thunks —
# focused coverage for SSPGVD/DSPGVD with packed row-major A/B reorder, queried
# WORK/IWORK allocation, and optional row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_spgvd_family
    test_conv_thunks_spgvd_family.c
)

target_include_directories(test_conv_thunks_spgvd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_spgvd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_spgvd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_spgvd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_spgvd_family COMMAND test_conv_thunks_spgvd_family)

message(STATUS "Packed symmetric generalized divide-and-conquer convention thunk test enabled")

# ============================================================================
# Generalized symmetric/Hermitian eigenproblem convention thunks — focused
# synthetic coverage for SYGV/HEGV and SYGVD/HEGVD query and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_generalized_eigen_family
    test_conv_thunks_generalized_eigen_family.c
)

target_include_directories(test_conv_thunks_generalized_eigen_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_generalized_eigen_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_generalized_eigen_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_generalized_eigen_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_generalized_eigen_family COMMAND test_conv_thunks_generalized_eigen_family)

message(STATUS "Generalized symmetric/Hermitian eigenproblem convention thunk test enabled")

# ============================================================================
# Least-squares driver convention thunks — focused synthetic coverage for the
# GELS quartet query and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_gels_family
    test_conv_thunks_gels_family.c
)

target_include_directories(test_conv_thunks_gels_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gels_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gels_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gels_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gels_family COMMAND test_conv_thunks_gels_family)

message(STATUS "Least-squares driver convention thunk test enabled")

# ============================================================================
# General band matrix-vector convention thunks — focused synthetic coverage for
# the GBMVX quartet's layout-aware band translation and reverse bridges.
# ============================================================================
add_executable(test_conv_thunks_gbmvx_family
    test_conv_thunks_gbmvx_family.c
)

target_include_directories(test_conv_thunks_gbmvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbmvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbmvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbmvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbmvx_family COMMAND test_conv_thunks_gbmvx_family)

message(STATUS "General band matrix-vector convention thunk test enabled")

# ============================================================================
# RZ Apply Convention Thunks — direct synthetic coverage for ORMR3/UNMR3
# bridges with side/trans/l metadata and row-major A/C translation.
# ============================================================================
add_executable(test_conv_thunks_rz_apply_family
    test_conv_thunks_rz_apply_family.c
)

target_include_directories(test_conv_thunks_rz_apply_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_rz_apply_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_rz_apply_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_rz_apply_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_rz_apply_family COMMAND test_conv_thunks_rz_apply_family)

message(STATUS "RZ apply convention thunk test enabled")

# ============================================================================
# Positive Band Equilibration Convention Thunks — synthetic coverage for
# PBEQU bridges with band-storage translation and real scale outputs.
# ============================================================================
add_executable(test_conv_thunks_pb_equ_family
    test_conv_thunks_pb_equ_family.c
)

target_include_directories(test_conv_thunks_pb_equ_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pb_equ_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pb_equ_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pb_equ_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pb_equ_family COMMAND test_conv_thunks_pb_equ_family)

message(STATUS "Positive band equilibration convention thunk test enabled")

# ============================================================================
# General Band Equilibration Convention Thunks — synthetic coverage for GBEQU
# bridges with row-major general-band translation and real scalar outputs.
# ============================================================================
add_executable(test_conv_thunks_gbequ_family
    test_conv_thunks_gbequ_family.c
)

target_include_directories(test_conv_thunks_gbequ_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbequ_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbequ_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbequ_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbequ_family COMMAND test_conv_thunks_gbequ_family)

message(STATUS "General band equilibration convention thunk test enabled")

# ============================================================================
# General Band Factorization Convention Thunks — synthetic coverage for
# GBTRF bridges with expanded factor-storage translation and row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_gbtrf_family
    test_conv_thunks_gbtrf_family.c
)

target_include_directories(test_conv_thunks_gbtrf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbtrf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbtrf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbtrf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbtrf_family COMMAND test_conv_thunks_gbtrf_family)

message(STATUS "General band factorization convention thunk test enabled")

# ============================================================================
# General Band Factored Solve Convention Thunks — synthetic coverage for
# GBTRS bridges with expanded factor-storage translation and RHS copy-back.
# ============================================================================
add_executable(test_conv_thunks_gbtrs_family
    test_conv_thunks_gbtrs_family.c
)

target_include_directories(test_conv_thunks_gbtrs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbtrs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbtrs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbtrs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbtrs_family COMMAND test_conv_thunks_gbtrs_family)

message(STATUS "General band factored solve convention thunk test enabled")

# ============================================================================
# General Band Direct Solve Convention Thunks — synthetic coverage for
# GBSV bridges with expanded band-storage translation and in-place copy-back.
# ============================================================================
add_executable(test_conv_thunks_gbsv_family
    test_conv_thunks_gbsv_family.c
)

target_include_directories(test_conv_thunks_gbsv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbsv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbsv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbsv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbsv_family COMMAND test_conv_thunks_gbsv_family)

message(STATUS "General band direct solve convention thunk test enabled")

# ============================================================================
# Positive Band Condition-Estimation Convention Thunks — synthetic coverage
# for PBCON bridges with band-storage translation and fixed internal scratch.
# ============================================================================
add_executable(test_conv_thunks_pb_condition_family
    test_conv_thunks_pb_condition_family.c
)

target_include_directories(test_conv_thunks_pb_condition_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pb_condition_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pb_condition_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pb_condition_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pb_condition_family COMMAND test_conv_thunks_pb_condition_family)

message(STATUS "Positive band condition-estimation convention thunk test enabled")

# ============================================================================
# General Band Condition-Estimate Convention Thunks — synthetic coverage for
# GBCON bridges with expanded GBTRF factor-storage translation.
# ============================================================================
add_executable(test_conv_thunks_gbcon_family
    test_conv_thunks_gbcon_family.c
)

target_include_directories(test_conv_thunks_gbcon_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbcon_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbcon_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbcon_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbcon_family COMMAND test_conv_thunks_gbcon_family)

message(STATUS "General band condition-estimation convention thunk test enabled")

# ============================================================================
# Positive Band Solve Convention Thunks — synthetic coverage for PBSV bridges
# with band-storage translation, RHS transpose, and row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_pb_solve_family
    test_conv_thunks_pb_solve_family.c
)

target_include_directories(test_conv_thunks_pb_solve_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pb_solve_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pb_solve_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pb_solve_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pb_solve_family COMMAND test_conv_thunks_pb_solve_family)

message(STATUS "Positive band solve convention thunk test enabled")

# ============================================================================
# Positive Band Expert Driver Convention Thunks — synthetic coverage for PBSVX
# bridges with FACT/EQUED policy, band-storage translation, and selective
# copy-back.
# ============================================================================
add_executable(test_conv_thunks_pb_svx_family
    test_conv_thunks_pb_svx_family.c
)

target_include_directories(test_conv_thunks_pb_svx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pb_svx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pb_svx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pb_svx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pb_svx_family COMMAND test_conv_thunks_pb_svx_family)

message(STATUS "Positive band expert-driver convention thunk test enabled")

# ============================================================================
# General Band Expert Driver Convention Thunks — synthetic coverage for GBSVX
# bridges with compact-AB expansion, factored-band translation, and selective
# row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_gbsvx_family
    test_conv_thunks_gbsvx_family.c
)

target_include_directories(test_conv_thunks_gbsvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gbsvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gbsvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gbsvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gbsvx_family COMMAND test_conv_thunks_gbsvx_family)

message(STATUS "General band expert-driver convention thunk test enabled")

# ============================================================================
# Tridiagonal Expert Driver Convention Thunks — synthetic coverage for GTSVX
# bridges with compact LAPACKE ABI, hidden workspace allocation, and complex
# WORK/RWORK handling.
# ============================================================================
add_executable(test_conv_thunks_gtsvx_family
    test_conv_thunks_gtsvx_family.c
)

target_include_directories(test_conv_thunks_gtsvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_gtsvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_gtsvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_gtsvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_gtsvx_family COMMAND test_conv_thunks_gtsvx_family)

message(STATUS "Tridiagonal expert-driver convention thunk test enabled")

# ============================================================================
# Positive Band Factor/Factored-Solve Convention Thunks — synthetic coverage
# for PBTRF/PBTRS bridges with band-storage translation and selective copy-back.
# ============================================================================
add_executable(test_conv_thunks_pb_factor_family
    test_conv_thunks_pb_factor_family.c
)

target_include_directories(test_conv_thunks_pb_factor_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pb_factor_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pb_factor_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pb_factor_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pb_factor_family COMMAND test_conv_thunks_pb_factor_family)

message(STATUS "Positive band factor/factored-solve convention thunk test enabled")

# ============================================================================
# Symmetric Band Factor Convention Thunks — focused synthetic coverage for
# SSBTRF/DSBTRF using compact symmetric-band row-major translation and copy-back.
# ============================================================================
add_executable(test_conv_thunks_sbtrf_family
    test_conv_thunks_sbtrf_family.c
)

target_include_directories(test_conv_thunks_sbtrf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbtrf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbtrf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbtrf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbtrf_family COMMAND test_conv_thunks_sbtrf_family)

message(STATUS "Symmetric band factor convention thunk test enabled")

# ============================================================================
# Positive Packed Factor Convention Thunks — synthetic coverage for PPTRF
# bridges with packed-storage translation and row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_pptrf_family
    test_conv_thunks_pptrf_family.c
)

target_include_directories(test_conv_thunks_pptrf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pptrf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pptrf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pptrf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pptrf_family COMMAND test_conv_thunks_pptrf_family)

message(STATUS "Positive packed factor convention thunk test enabled")

# ============================================================================
# Positive Packed Inverse Convention Thunks — synthetic coverage for PPTRI
# bridges with packed-storage translation and row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_pptri_family
    test_conv_thunks_pptri_family.c
)

target_include_directories(test_conv_thunks_pptri_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pptri_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pptri_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pptri_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pptri_family COMMAND test_conv_thunks_pptri_family)

# ============================================================================
# Hermitian Packed Inverse Convention Thunks — synthetic coverage for HPTRI
# complex variants using packed row-major reorder and info propagation.
# ============================================================================
add_executable(test_conv_thunks_hptri_family
    test_conv_thunks_hptri_family.c
)

target_include_directories(test_conv_thunks_hptri_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hptri_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hptri_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hptri_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hptri_family COMMAND test_conv_thunks_hptri_family)

# ============================================================================
# Hermitian Packed Factor Convention Thunks — synthetic coverage for HPTRF
# complex variants using packed row-major reorder and info propagation.
# ============================================================================
add_executable(test_conv_thunks_hptrf_family
    test_conv_thunks_hptrf_family.c
)

target_include_directories(test_conv_thunks_hptrf_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hptrf_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hptrf_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hptrf_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

message(STATUS "Positive packed inverse convention thunk test enabled")

# ============================================================================
# Positive Packed Factored-Solve Convention Thunks — synthetic coverage for
# PPTRS bridges with packed-storage translation and RHS row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_pptrs_family
    test_conv_thunks_pptrs_family.c
)

target_include_directories(test_conv_thunks_pptrs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pptrs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pptrs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pptrs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pptrs_family COMMAND test_conv_thunks_pptrs_family)

message(STATUS "Positive packed factored-solve convention thunk test enabled")

# ============================================================================
# Positive Tridiagonal Condition Convention Thunks — synthetic coverage for
# PTCON bridges with hidden work and real/complex off-diagonal forwarding.
# ============================================================================
add_executable(test_conv_thunks_ptcon_family
    test_conv_thunks_ptcon_family.c
)

target_include_directories(test_conv_thunks_ptcon_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ptcon_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ptcon_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ptcon_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ptcon_family COMMAND test_conv_thunks_ptcon_family)

message(STATUS "Positive tridiagonal condition convention thunk test enabled")

# ============================================================================
# Positive Tridiagonal QR Iteration Convention Thunks — focused coverage for
# SPTEQR bridges with fixed workspace and compz-sensitive row-major handling.
# ============================================================================
add_executable(test_conv_thunks_pteqr_family
    test_conv_thunks_pteqr_family.c
)

target_include_directories(test_conv_thunks_pteqr_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pteqr_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pteqr_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pteqr_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pteqr_family COMMAND test_conv_thunks_pteqr_family)

message(STATUS "Positive tridiagonal QR iteration convention thunk test enabled")

# ============================================================================
# Positive Tridiagonal Factor/Factored-Solve Convention Thunks — focused
# coverage for PTTRF/PTTRS bridges with vector forwarding and RHS copy-back.
# ============================================================================
add_executable(test_conv_thunks_pt_factor_family
    test_conv_thunks_pt_factor_family.c
)

target_include_directories(test_conv_thunks_pt_factor_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pt_factor_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pt_factor_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pt_factor_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pt_factor_family COMMAND test_conv_thunks_pt_factor_family)

message(STATUS "Positive tridiagonal factor/factored-solve convention thunk test enabled")

# ============================================================================
# Positive Tridiagonal Direct-Solve Convention Thunks — focused coverage for
# PTSV bridges with in-place D/E mutation and RHS row-major copy-back.
# ============================================================================
add_executable(test_conv_thunks_ptsv_family
    test_conv_thunks_ptsv_family.c
)

target_include_directories(test_conv_thunks_ptsv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ptsv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ptsv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ptsv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ptsv_family COMMAND test_conv_thunks_ptsv_family)

message(STATUS "Positive tridiagonal direct-solve convention thunk test enabled")

# ============================================================================
# Positive Tridiagonal Expert-Driver Convention Thunks — focused coverage for
# PTSVX bridges with FACT policy, direct factor vectors, and X-only copy-back.
# ============================================================================
add_executable(test_conv_thunks_ptsvx_family
    test_conv_thunks_ptsvx_family.c
)

target_include_directories(test_conv_thunks_ptsvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ptsvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ptsvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ptsvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ptsvx_family COMMAND test_conv_thunks_ptsvx_family)

message(STATUS "Positive tridiagonal expert-driver convention thunk test enabled")

# ============================================================================
# PTRFS Split Convention Thunks — focused coverage for the repo-local split
# boundary where real PTRFS uses packed triangular refinement ABI and complex
# PTRFS uses positive-tridiagonal refinement ABI.
# ============================================================================
add_executable(test_conv_thunks_ptrfs_family
    test_conv_thunks_ptrfs_family.c
)

target_include_directories(test_conv_thunks_ptrfs_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_ptrfs_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_ptrfs_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_ptrfs_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_ptrfs_family COMMAND test_conv_thunks_ptrfs_family)

message(STATUS "PTRFS split convention thunk test enabled")

# ============================================================================
# Complex ROTG Convention Thunks — focused coverage for CROTG/ZROTG scalar
# pass-through bridges with identical pointer-based CBLAS and Fortran ABIs.
# ============================================================================
add_executable(test_conv_thunks_rotg_complex_family
    test_conv_thunks_rotg_complex_family.c
)

target_include_directories(test_conv_thunks_rotg_complex_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_rotg_complex_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_rotg_complex_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_rotg_complex_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_rotg_complex_family COMMAND test_conv_thunks_rotg_complex_family)

message(STATUS "Complex ROTG convention thunk test enabled")

# ============================================================================
# Reciprocal Scaling Convention Thunks — focused coverage for SRSCL/DRSCL and
# the complex auxiliary variants CSRSCL/ZDRSCL.
# ============================================================================
add_executable(test_conv_thunks_rscl_family
    test_conv_thunks_rscl_family.c
)

target_include_directories(test_conv_thunks_rscl_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_rscl_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_rscl_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_rscl_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_rscl_family COMMAND test_conv_thunks_rscl_family)

message(STATUS "RSCL convention thunk test enabled")

# ============================================================================
# Symmetric Banded Eigen Convention Thunks — focused coverage for SSBEV/DSBEV
# with fixed hidden work, symmetric-band row-major translation, and optional
# row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_sbev_family
    test_conv_thunks_sbev_family.c
)

target_include_directories(test_conv_thunks_sbev_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbev_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbev_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbev_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbev_family COMMAND test_conv_thunks_sbev_family)

message(STATUS "Symmetric banded eigen convention thunk test enabled")

# ============================================================================
# Symmetric Banded Divide-and-Conquer Eigen Convention Thunks — focused
# coverage for SSBEVD/DSBEVD with hidden WORK/IWORK allocation, symmetric-band
# row-major translation, and optional row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_sbevd_family
    test_conv_thunks_sbevd_family.c
)

target_include_directories(test_conv_thunks_sbevd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbevd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbevd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbevd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbevd_family COMMAND test_conv_thunks_sbevd_family)

message(STATUS "Symmetric banded divide-and-conquer convention thunk test enabled")

# ============================================================================
# Symmetric Banded Selection Eigen Convention Thunks — focused coverage for
# SSBEVX/DSBEVX with hidden WORK/IWORK allocation, AB/Q/Z row-major copy-back,
# and selection-output forwarding.
# ============================================================================
add_executable(test_conv_thunks_sbevx_family
    test_conv_thunks_sbevx_family.c
)

target_include_directories(test_conv_thunks_sbevx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbevx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbevx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbevx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbevx_family COMMAND test_conv_thunks_sbevx_family)

message(STATUS "Symmetric banded selection convention thunk test enabled")

# ============================================================================
# Hermitian Banded Eigen Convention Thunks — focused coverage for CHBEV/ZHBEV
# with hidden WORK/RWORK allocation, Hermitian band translation, and
# eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hbev_family
    test_conv_thunks_hbev_family.c
)

target_include_directories(test_conv_thunks_hbev_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hbev_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hbev_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hbev_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hbev_family COMMAND test_conv_thunks_hbev_family)

message(STATUS "Hermitian banded convention thunk test enabled")

# ============================================================================
# Hermitian Banded EVD Convention Thunks - focused coverage for CHBEVD/ZHBEVD
# with hidden WORK/RWORK/IWORK query-allocation, Hermitian band translation,
# and eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hbevd_family
    test_conv_thunks_hbevd_family.c
)

target_include_directories(test_conv_thunks_hbevd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hbevd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hbevd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hbevd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hbevd_family COMMAND test_conv_thunks_hbevd_family)

message(STATUS "Hermitian banded EVD convention thunk test enabled")

# ============================================================================
# Vtable Compatibility Autofill (Pilot) - validates mathematically compatible
# slot propagation (e.g., CHBEVD -> CHBEV) and overwrite-precedence rules.
# ============================================================================
add_executable(test_vtable_compat_autofill_family
    test_vtable_compat_autofill_family.c
)

target_include_directories(test_vtable_compat_autofill_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_vtable_compat_autofill_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_vtable_compat_autofill_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_vtable_compat_autofill_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME vtable_compat_autofill_family COMMAND test_vtable_compat_autofill_family)

message(STATUS "Vtable compatibility autofill pilot test enabled")

# ============================================================================
# Vtable Compatibility Autofill (Generalized Hermitian Alias Families)
# ============================================================================
add_executable(test_vtable_compat_autofill_generalized_hermitian_family
    test_vtable_compat_autofill_generalized_hermitian_family.c
)

target_include_directories(test_vtable_compat_autofill_generalized_hermitian_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_vtable_compat_autofill_generalized_hermitian_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_vtable_compat_autofill_generalized_hermitian_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_vtable_compat_autofill_generalized_hermitian_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME vtable_compat_autofill_generalized_hermitian_family
         COMMAND test_vtable_compat_autofill_generalized_hermitian_family)

message(STATUS "Vtable compatibility generalized-Hermitian alias test enabled")

# ============================================================================
# Symmetric Generalized Banded Eigen Convention Thunks — focused coverage for
# SSBGV/DSBGV with hidden WORK allocation, A/B row-major band translation, and
# optional row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_sbgv_family
    test_conv_thunks_sbgv_family.c
)

target_include_directories(test_conv_thunks_sbgv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbgv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbgv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbgv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbgv_family COMMAND test_conv_thunks_sbgv_family)

message(STATUS "Symmetric generalized banded convention thunk test enabled")

# ============================================================================
# Hermitian Generalized Banded Eigen Convention Thunks — focused coverage for
# CHBGV/ZHBGV with hidden WORK/RWORK allocation, A/B row-major band
# translation, and eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hbgv_family
    test_conv_thunks_hbgv_family.c
)

target_include_directories(test_conv_thunks_hbgv_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hbgv_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hbgv_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hbgv_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hbgv_family COMMAND test_conv_thunks_hbgv_family)

message(STATUS "Hermitian generalized banded convention thunk test enabled")

# ============================================================================
# Hermitian Generalized Banded Divide-and-Conquer Eigen Convention Thunks —
# focused coverage for CHBGVD/ZHBGVD with query-based WORK/RWORK/IWORK
# allocation, A/B row-major band translation, and eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hbgvd_family
    test_conv_thunks_hbgvd_family.c
)

target_include_directories(test_conv_thunks_hbgvd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hbgvd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hbgvd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hbgvd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hbgvd_family COMMAND test_conv_thunks_hbgvd_family)

# ============================================================================
# Hermitian Generalized Banded Selection Eigen Convention Thunks — focused
# coverage for CHBGVX/ZHBGVX with hidden WORK/RWORK/IWORK allocation, A/B/Q
# row-major band translation, and selected eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_hbgvx_family
    test_conv_thunks_hbgvx_family.c
)

target_include_directories(test_conv_thunks_hbgvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_hbgvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_hbgvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_hbgvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_hbgvx_family COMMAND test_conv_thunks_hbgvx_family)

# ============================================================================
# Symmetric Generalized Banded Divide-and-Conquer Eigen Convention Thunks —
# focused coverage for SSBGVD/DSBGVD with hidden WORK/IWORK allocation,
# A/B row-major band translation, and optional row-major eigenvector copy-back.
# ============================================================================
add_executable(test_conv_thunks_sbgvd_family
    test_conv_thunks_sbgvd_family.c
)

target_include_directories(test_conv_thunks_sbgvd_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbgvd_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbgvd_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbgvd_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbgvd_family COMMAND test_conv_thunks_sbgvd_family)

message(STATUS "Symmetric generalized banded divide-and-conquer convention thunk test enabled")

# ============================================================================
# Symmetric Generalized Banded Selection Eigen Convention Thunks — focused
# coverage for SSBGVX/DSBGVX with hidden WORK/IWORK allocation, A/B/Q/Z
# row-major copy-back, and selection-output forwarding.
# ============================================================================
add_executable(test_conv_thunks_sbgvx_family
    test_conv_thunks_sbgvx_family.c
)

target_include_directories(test_conv_thunks_sbgvx_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_sbgvx_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_sbgvx_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_sbgvx_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_sbgvx_family COMMAND test_conv_thunks_sbgvx_family)

message(STATUS "Symmetric generalized banded selection convention thunk test enabled")

# ============================================================================
# Positive Band Refinement Convention Thunks — synthetic coverage for PBRFS
# bridges with band-storage translation, hidden scratch, and X-only copy-back.
# ============================================================================
add_executable(test_conv_thunks_pb_refine_family
    test_conv_thunks_pb_refine_family.c
)

target_include_directories(test_conv_thunks_pb_refine_family PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_pb_refine_family faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_pb_refine_family POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_pb_refine_family>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_pb_refine_family COMMAND test_conv_thunks_pb_refine_family)

message(STATUS "Positive band refinement convention thunk test enabled")

# ============================================================================
# LAPACK Convention Thunk Policy — synthetic coverage for layout and mode-bit
# preservation in generated LAPACK C bridges.
# ============================================================================
add_executable(test_conv_thunks_lapack_policy
    test_conv_thunks_lapack_policy.c
)

target_include_directories(test_conv_thunks_lapack_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_lapack_policy faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_lapack_policy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_lapack_policy>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_lapack_policy COMMAND test_conv_thunks_lapack_policy)

message(STATUS "LAPACK convention thunk policy test enabled")

# ============================================================================
# Judge GELSD Complex Regression — compare the judge CGELSD/ZGELSD path against
# direct reference DLL exports on the generated small corpus.
# ============================================================================
add_executable(test_judge_gelsd_complex_regression
    test_judge_gelsd_complex_regression.c
)

target_include_directories(test_judge_gelsd_complex_regression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(test_judge_gelsd_complex_regression PRIVATE
    FB_REFERENCE_DLL_DIR="${CMAKE_SOURCE_DIR}/../faster-blaster-reference/build-extended"
)

target_link_libraries(test_judge_gelsd_complex_regression faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_judge_gelsd_complex_regression POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_judge_gelsd_complex_regression>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME judge_gelsd_complex_regression COMMAND test_judge_gelsd_complex_regression)

message(STATUS "Judge GELSD complex regression test enabled")

# ============================================================================
# Judge GEMM3M Regression — exercise the complex alias/plain batch boundary
# through fb_judge_run() so ABI mismatches do not regress silently.
# ============================================================================
add_executable(test_judge_gemm3m_regression
    test_judge_gemm3m_regression.c
)

target_include_directories(test_judge_gemm3m_regression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(test_judge_gemm3m_regression PRIVATE
    FB_REFERENCE_DLL_DIR="${CMAKE_SOURCE_DIR}/../faster-blaster-reference/build-extended"
)

target_link_libraries(test_judge_gemm3m_regression faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_judge_gemm3m_regression POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_judge_gemm3m_regression>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME judge_gemm3m_regression COMMAND test_judge_gemm3m_regression)

message(STATUS "Judge GEMM3M regression test enabled")

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
        ${CMAKE_SOURCE_DIR}/src/core/backend_auto_detect.c
        ${CMAKE_SOURCE_DIR}/src/core/op_vtable_map.c
        ${CMAKE_SOURCE_DIR}/src/backends/sym_tables/lapacke_sym_table.c
    )

    target_include_directories(openblas_backend_lib PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    # OpenBLAS test executable
    add_executable(test_openblas_backend
        test_openblas_backend.c
        ${CMAKE_SOURCE_DIR}/src/core/op_vtable_map.c
    )

    target_link_libraries(test_openblas_backend openblas_backend_lib)

    if(UNIX)
        target_link_libraries(test_openblas_backend m dl pthread)
    endif()

    add_test(NAME OpenBLAS_Backend_Tests COMMAND test_openblas_backend)

    if(WIN32)
        set_tests_properties(OpenBLAS_Backend_Tests PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:faster-blaster>;PATH=path_list_prepend:${CMAKE_BINARY_DIR}/backends-install/openblas/bin")
    endif()

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
# Cross-Op Convention Thunk Coverage — verifies fb_install_conv_thunks wiring
# against the global thunk tables for every operation id in both directions.
# ============================================================================
add_executable(test_conv_thunks_cross_op_coverage
    test_conv_thunks_cross_op_coverage.c
)

target_include_directories(test_conv_thunks_cross_op_coverage PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_conv_thunks_cross_op_coverage faster-blaster)

if(WIN32)
    add_custom_command(TARGET test_conv_thunks_cross_op_coverage POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:faster-blaster>
            $<TARGET_FILE_DIR:test_conv_thunks_cross_op_coverage>
        COMMENT "Copying faster-blaster.dll to test directory"
    )
endif()

add_test(NAME conv_thunks_cross_op_coverage COMMAND test_conv_thunks_cross_op_coverage)

message(STATUS "Cross-op convention thunk coverage test enabled")

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


