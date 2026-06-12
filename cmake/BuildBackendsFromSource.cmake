# BuildBackendsFromSource.cmake
# Automatically build BLAS/LAPACK backends from source with hardware-optimized flags
#
# This module creates a superbuild that compiles backend libraries with:
# - CPU-specific optimizations (Zen2/3/4, Intel Skylake/Ice Lake, etc.)
# - GPU architecture targeting (CUDA compute capability, ROCm gfx arch)
# - Threading optimizations (OpenMP, pthreads)
# - Maximum performance compiler flags

include(ExternalProject)
include(ProcessorCount)

# Options for controlling what gets built
option(FB_BUILD_BACKENDS_FROM_SOURCE "Build all backends from source for max performance" ON)
option(FB_BUILD_BLIS_FROM_SOURCE "Build BLIS from source" ON)
# OpenBLAS: Now builds successfully on Windows using clang-cl (C99/VLA support + MSVC ABI)
option(FB_BUILD_OPENBLAS_FROM_SOURCE "Build OpenBLAS from source with clang-cl" ON)
option(FB_BUILD_CLBLAST_FROM_SOURCE "Build CLBlast from source" ON)

if(WIN32)
    option(FB_USE_PREBUILT_OPENBLAS_WINDOWS
        "Use precompiled OpenBLAS binaries on Windows instead of building OpenBLAS from source"
        ON)
    option(FB_OPENBLAS_WINDOWS_PREBUILT_DOWNLOAD
        "Automatically download official precompiled OpenBLAS binaries on Windows"
        ON)
    set(FB_OPENBLAS_WINDOWS_PREBUILT_VERSION "0.3.30" CACHE STRING
        "OpenBLAS release version for precompiled Windows package download")
    set(FB_OPENBLAS_WINDOWS_PREBUILT_ROOT "C:/libraries/OpenBLAS-0.3.30-x64" CACHE PATH
        "Path to extracted precompiled OpenBLAS root (must contain include/ and lib/)" FORCE)
endif()

# OxiBLAS: Pure-Rust BLAS/LAPACK via oxiblas-ffi.
# Upstream retired oxiblas-ffi in v0.2.0+, so the auto-clone uses the last
# FFI-capable release by default. Override with
#   -DFB_BUILD_OXIBLAS_FROM_SOURCE=ON|OFF
# Set OXIBLAS_CARGO_SOURCE_DIR to an existing checkout to skip the git clone.
find_program(CARGO_EXECUTABLE cargo DOC "Rust cargo package manager (https://rustup.rs)")
find_package(Git QUIET)
if(CARGO_EXECUTABLE AND Git_FOUND)
    message(STATUS "cargo and git found — OxiBLAS from-source build ON by default")
    set(_oxiblas_default ON)
else()
    if(NOT CARGO_EXECUTABLE)
        message(STATUS "cargo not found — OxiBLAS from-source disabled (https://rustup.rs)")
    elseif(NOT Git_FOUND)
        message(STATUS "git not found — OxiBLAS from-source disabled")
    endif()
    set(_oxiblas_default OFF)
endif()
option(FB_BUILD_OXIBLAS_FROM_SOURCE
    "Build OxiBLAS FFI from source (cargo build -p oxiblas-ffi, requires Rust 1.85+)"
    ${_oxiblas_default})
set(OXIBLAS_CARGO_SOURCE_DIR "" CACHE PATH
    "Path to oxiblas repository root (leave empty to auto-clone from GitHub)")
set(FB_OXIBLAS_FFI_GIT_TAG "v0.1.1" CACHE STRING
    "OxiBLAS git tag/branch containing the oxiblas-ffi workspace member")

# Installation prefix for built libraries
set(FB_BACKENDS_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/backends-install" CACHE PATH 
    "Installation prefix for source-built backends")

# Get number of cores for parallel compilation
ProcessorCount(NUM_CORES)
if(NUM_CORES EQUAL 0)
    set(NUM_CORES 4)
endif()

# OOM guardrail: cap backend parallelism unless explicitly overridden.
set(FB_BACKEND_BUILD_JOBS "" CACHE STRING
    "Max parallel jobs for backend source builds (empty = auto cap)")

if(FB_BACKEND_BUILD_JOBS STREQUAL "")
    if(WIN32)
        # Windows clang-cl + large backend builds can spike memory usage.
        if(NUM_CORES GREATER 8)
            set(FB_EFFECTIVE_BACKEND_JOBS 8)
        else()
            set(FB_EFFECTIVE_BACKEND_JOBS ${NUM_CORES})
        endif()
    else()
        if(NUM_CORES GREATER 16)
            set(FB_EFFECTIVE_BACKEND_JOBS 16)
        else()
            set(FB_EFFECTIVE_BACKEND_JOBS ${NUM_CORES})
        endif()
    endif()
else()
    set(FB_EFFECTIVE_BACKEND_JOBS ${FB_BACKEND_BUILD_JOBS})
endif()

set(NUM_CORES ${FB_EFFECTIVE_BACKEND_JOBS})
message(STATUS "Building backends with ${NUM_CORES} parallel jobs")

# Function to convert Windows paths to WSL paths
function(win_path_to_wsl_path WIN_PATH OUT_VAR)
    if(WIN32 AND WIN_PATH MATCHES "^([A-Za-z]):(.*)")
        string(TOLOWER "${CMAKE_MATCH_1}" drive_lower)
        string(REPLACE "\\" "/" unix_style "${CMAKE_MATCH_2}")
        set(${OUT_VAR} "/mnt/${drive_lower}${unix_style}" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "${WIN_PATH}" PARENT_SCOPE)
    endif()
endfunction()

#==============================================================================
# HARDWARE DETECTION
#==============================================================================

# Detect CPU architecture and vendor
function(detect_cpu_architecture OUT_VENDOR OUT_ARCH OUT_CONFIG)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x86")
        if(WIN32)
            execute_process(
                COMMAND powershell -NoProfile -Command 
                    "Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name"
                OUTPUT_VARIABLE CPU_INFO
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
        elseif(UNIX)
            execute_process(
                COMMAND sh -c "grep 'model name' /proc/cpuinfo | head -1"
                OUTPUT_VARIABLE CPU_INFO
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
        endif()
        
        string(STRIP "${CPU_INFO}" CPU_INFO)
        
        # AMD CPUs
        if(CPU_INFO MATCHES "AMD")
            set(${OUT_VENDOR} "AMD" PARENT_SCOPE)
            
            # Detect specific Zen architecture
            if(CPU_INFO MATCHES "Ryzen.*(9000|8000|7[5-9][0-9][0-9])")
                set(${OUT_ARCH} "zen4" PARENT_SCOPE)
                set(${OUT_CONFIG} "znver4" PARENT_SCOPE)
                message(STATUS "Detected AMD Zen 4 CPU: ${CPU_INFO}")
            elseif(CPU_INFO MATCHES "Ryzen.*(7000|[5-6][0-9][0-9][0-9])|EPYC.*(9[0-9][0-9][0-9])")
                set(${OUT_ARCH} "zen4" PARENT_SCOPE)
                set(${OUT_CONFIG} "znver4" PARENT_SCOPE)
                message(STATUS "Detected AMD Zen 4 CPU: ${CPU_INFO}")
            elseif(CPU_INFO MATCHES "Ryzen.*(5000|[3-4][0-9][0-9][0-9])|EPYC.*(7[0-9][0-9][0-9])")
                set(${OUT_ARCH} "zen3" PARENT_SCOPE)
                set(${OUT_CONFIG} "znver3" PARENT_SCOPE)
                message(STATUS "Detected AMD Zen 3 CPU: ${CPU_INFO}")
            elseif(CPU_INFO MATCHES "Ryzen.*(3000|[1-2][0-9][0-9][0-9])")
                set(${OUT_ARCH} "zen2" PARENT_SCOPE)
                set(${OUT_CONFIG} "znver2" PARENT_SCOPE)
                message(STATUS "Detected AMD Zen 2 CPU: ${CPU_INFO}")
            else()
                set(${OUT_ARCH} "zen" PARENT_SCOPE)
                set(${OUT_CONFIG} "znver1" PARENT_SCOPE)
                message(STATUS "Detected AMD Zen CPU: ${CPU_INFO}")
            endif()
            
        # Intel CPUs
        elseif(CPU_INFO MATCHES "Intel")
            set(${OUT_VENDOR} "Intel" PARENT_SCOPE)
            
            if(CPU_INFO MATCHES "Sapphire Rapids|SPR")
                set(${OUT_ARCH} "sapphirerapids" PARENT_SCOPE)
                set(${OUT_CONFIG} "sapphirerapids" PARENT_SCOPE)
                message(STATUS "Detected Intel Sapphire Rapids CPU: ${CPU_INFO}")
            elseif(CPU_INFO MATCHES "Ice Lake|ICL")
                set(${OUT_ARCH} "icelake" PARENT_SCOPE)
                set(${OUT_CONFIG} "icelake-server" PARENT_SCOPE)
                message(STATUS "Detected Intel Ice Lake CPU: ${CPU_INFO}")
            elseif(CPU_INFO MATCHES "Skylake|SKL")
                set(${OUT_ARCH} "skylake" PARENT_SCOPE)
                set(${OUT_CONFIG} "skylake-avx512" PARENT_SCOPE)
                message(STATUS "Detected Intel Skylake CPU: ${CPU_INFO}")
            elseif(CPU_INFO MATCHES "Cascade Lake|CLX")
                set(${OUT_ARCH} "cascadelake" PARENT_SCOPE)
                set(${OUT_CONFIG} "cascadelake" PARENT_SCOPE)
                message(STATUS "Detected Intel Cascade Lake CPU: ${CPU_INFO}")
            else()
                set(${OUT_ARCH} "haswell" PARENT_SCOPE)
                set(${OUT_CONFIG} "haswell" PARENT_SCOPE)
                message(STATUS "Detected Intel CPU (generic): ${CPU_INFO}")
            endif()
        else()
            set(${OUT_VENDOR} "Generic" PARENT_SCOPE)
            set(${OUT_ARCH} "generic" PARENT_SCOPE)
            set(${OUT_CONFIG} "generic" PARENT_SCOPE)
            message(STATUS "Generic CPU detected: ${CPU_INFO}")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        set(${OUT_VENDOR} "ARM" PARENT_SCOPE)
        set(${OUT_ARCH} "armv8" PARENT_SCOPE)
        set(${OUT_CONFIG} "armv8.2-a" PARENT_SCOPE)
        message(STATUS "Detected ARM64 CPU")
    else()
        set(${OUT_VENDOR} "Generic" PARENT_SCOPE)
        set(${OUT_ARCH} "generic" PARENT_SCOPE)
        set(${OUT_CONFIG} "generic" PARENT_SCOPE)
    endif()
endfunction()

# Detect GPU architectures
function(detect_gpu_architectures OUT_HAS_NVIDIA OUT_CUDA_ARCH OUT_HAS_AMD OUT_ROCM_ARCH OUT_HAS_OPENCL)
    set(${OUT_HAS_NVIDIA} FALSE PARENT_SCOPE)
    set(${OUT_CUDA_ARCH} "" PARENT_SCOPE)
    set(${OUT_HAS_AMD} FALSE PARENT_SCOPE)
    set(${OUT_ROCM_ARCH} "" PARENT_SCOPE)
    set(${OUT_HAS_OPENCL} FALSE PARENT_SCOPE)
    set(FB_DETECTED_CUDA_VERSION "" PARENT_SCOPE)
    
    # NVIDIA CUDA detection
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(_detected_cuda_arch "")
        set(${OUT_HAS_NVIDIA} TRUE PARENT_SCOPE)
        set(FB_DETECTED_CUDA_VERSION "${CUDAToolkit_VERSION}" PARENT_SCOPE)
        
        # Try to detect GPU compute capability
        if(CMAKE_CUDA_COMPILER)
            execute_process(
                COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
                OUTPUT_VARIABLE CUDA_ARCHS
                ERROR_QUIET
            )
            
            # Default to common architectures if detection fails
            if(CUDA_ARCHS)
                string(STRIP "${CUDA_ARCHS}" _detected_cuda_arch)
            else()
                # Common modern architectures: Pascal, Turing, Ampere, Ada, Hopper
                set(_detected_cuda_arch "60;61;70;75;80;86;89;90")
            endif()
        else()
            set(_detected_cuda_arch "60;70;75;80;86")
        endif()
        set(${OUT_CUDA_ARCH} "${_detected_cuda_arch}" PARENT_SCOPE)
        message(STATUS "NVIDIA CUDA detected - targeting architectures: ${_detected_cuda_arch}")
    endif()
    
    # AMD ROCm detection
    find_package(hip QUIET)
    if(hip_FOUND)
        set(${OUT_HAS_AMD} TRUE PARENT_SCOPE)
        
        # Detect ROCm GPU architecture
        execute_process(
            COMMAND rocminfo
            OUTPUT_VARIABLE ROCM_INFO
            ERROR_QUIET
        )
        
        if(ROCM_INFO MATCHES "gfx([0-9]+)")
            set(${OUT_ROCM_ARCH} "${CMAKE_MATCH_1}" PARENT_SCOPE)
            message(STATUS "AMD ROCm detected - gfx${CMAKE_MATCH_1}")
        else()
            # Common architectures: RDNA2, RDNA3, CDNA2, CDNA3
            set(${OUT_ROCM_ARCH} "gfx1030;gfx1100;gfx90a;gfx940" PARENT_SCOPE)
            message(STATUS "AMD ROCm detected - using default architectures")
        endif()
    endif()
    
    # OpenCL detection
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(${OUT_HAS_OPENCL} TRUE PARENT_SCOPE)
        message(STATUS "OpenCL detected: ${OpenCL_VERSION_STRING}")
    endif()
endfunction()

#==============================================================================
# COMPILER FLAGS OPTIMIZATION
#==============================================================================

function(get_optimized_compiler_flags CPU_VENDOR CPU_CONFIG OUT_C_FLAGS OUT_CXX_FLAGS)
    set(BASE_FLAGS "")
    set(ARCH_FLAGS "")
    
    if(MSVC)
        # MSVC flags
        set(BASE_FLAGS "/O2 /Ob2 /Oi /Ot /GL /GS- /fp:fast")
        
        if(CPU_VENDOR STREQUAL "AMD")
            # AMD-specific MSVC optimizations
            set(ARCH_FLAGS "/arch:AVX2")
        elseif(CPU_VENDOR STREQUAL "Intel")
            set(ARCH_FLAGS "/arch:AVX2")
        endif()
    else()
        # GCC/Clang flags
        set(BASE_FLAGS "-O3 -ffast-math -funroll-loops -fomit-frame-pointer")
        
        if(CPU_VENDOR STREQUAL "AMD")
            set(ARCH_FLAGS "-march=${CPU_CONFIG} -mtune=${CPU_CONFIG}")
        elseif(CPU_VENDOR STREQUAL "Intel")
            set(ARCH_FLAGS "-march=${CPU_CONFIG} -mtune=${CPU_CONFIG}")
        elseif(CPU_VENDOR STREQUAL "ARM")
            set(ARCH_FLAGS "-march=${CPU_CONFIG} -mtune=native")
        else()
            set(ARCH_FLAGS "-march=native -mtune=native")
        endif()
    endif()
    
    set(${OUT_C_FLAGS} "${BASE_FLAGS} ${ARCH_FLAGS}" PARENT_SCOPE)
    set(${OUT_CXX_FLAGS} "${BASE_FLAGS} ${ARCH_FLAGS}" PARENT_SCOPE)
endfunction()

#==============================================================================
# BUILD BLIS FROM SOURCE
#==============================================================================

function(build_blis_from_source CPU_ARCH)
    if(NOT FB_BUILD_BLIS_FROM_SOURCE)
        return()
    endif()
    
    message(STATUS "Configuring BLIS build from source (${CPU_ARCH} optimized)")
    
    set(BLIS_SOURCE_DIR "${CMAKE_BINARY_DIR}/blis-src")
    set(BLIS_BINARY_DIR "${CMAKE_BINARY_DIR}/blis-build")
    set(BLIS_INSTALL_DIR "${FB_BACKENDS_INSTALL_PREFIX}/blis")
    
    # Platform-specific configuration
    if(WIN32)
        # Windows: Windows git clones with CRLF line endings, making the BLIS bash
        # configure shebang "bash\r" which WSL cannot execute (exit 127).
        # Solution: delegate the git clone itself to WSL's git so the repo arrives
        # with LF endings, then configure/build/install through WSL as normal.
        execute_process(
            COMMAND wsl echo "wsl_ok"
            RESULT_VARIABLE WSL_RESULT
            OUTPUT_VARIABLE WSL_OUTPUT
            ERROR_QUIET
            TIMEOUT 10
        )
        string(STRIP "${WSL_OUTPUT}" WSL_OUTPUT)
        if(NOT WSL_RESULT EQUAL 0 OR NOT WSL_OUTPUT STREQUAL "wsl_ok")
            message(STATUS "WSL not functional -- skipping BLIS from-source build. Using system-installed AOCL-BLIS instead.")
            return()
        endif()

        win_path_to_wsl_path("${BLIS_SOURCE_DIR}" BLIS_SOURCE_WSL)
        win_path_to_wsl_path("${BLIS_INSTALL_DIR}" BLIS_INSTALL_WSL)

        # Clone via WSL git (LF endings) — skip if already cloned (idempotent)
        set(BLIS_DOWNLOAD_CMD
            wsl bash -c
            "[ -d '${BLIS_SOURCE_WSL}/.git' ] || git clone --depth=1 https://github.com/amd/blis.git '${BLIS_SOURCE_WSL}'"
        )

        ExternalProject_Add(blis-backend
            DOWNLOAD_COMMAND  ${BLIS_DOWNLOAD_CMD}
            SOURCE_DIR        ${BLIS_SOURCE_DIR}
            CONFIGURE_COMMAND wsl bash -c "cd '${BLIS_SOURCE_WSL}' && ./configure --prefix='${BLIS_INSTALL_WSL}' --enable-cblas --enable-threading=openmp --enable-shared --enable-static ${CPU_ARCH}"
            BUILD_COMMAND     wsl bash -c "cd '${BLIS_SOURCE_WSL}' && make -j${NUM_CORES}"
            INSTALL_COMMAND   wsl bash -c "cd '${BLIS_SOURCE_WSL}' && make install"
            BUILD_IN_SOURCE   TRUE
            LOG_DOWNLOAD      TRUE
            LOG_CONFIGURE     TRUE
            LOG_BUILD         TRUE
            LOG_INSTALL       TRUE
        )
    else()
        ExternalProject_Add(blis-backend
            GIT_REPOSITORY https://github.com/amd/blis.git
            GIT_TAG        master
            GIT_SHALLOW    TRUE
            SOURCE_DIR        ${BLIS_SOURCE_DIR}
            CONFIGURE_COMMAND ./configure --prefix=${BLIS_INSTALL_DIR} --enable-cblas --enable-threading=openmp --enable-shared --enable-static ${CPU_ARCH}
            BUILD_COMMAND     make -j${NUM_CORES}
            INSTALL_COMMAND   make install
            BUILD_IN_SOURCE   TRUE
            LOG_DOWNLOAD      TRUE
            LOG_CONFIGURE     TRUE
            LOG_BUILD         TRUE
            LOG_INSTALL       TRUE
        )
    endif()
    
    # Export variables for main project
    set(BLIS_FOUND TRUE PARENT_SCOPE)
    set(BLIS_INCLUDE_DIRS "${BLIS_INSTALL_DIR}/include/blis" PARENT_SCOPE)
    set(BLIS_LIBRARIES "${BLIS_INSTALL_DIR}/lib/libblis-mt.a" PARENT_SCOPE)
    
    message(STATUS "BLIS will be built and installed to: ${BLIS_INSTALL_DIR}")
endfunction()

#==============================================================================
# BUILD OPENBLAS FROM SOURCE
#==============================================================================

function(build_openblas_from_source CPU_VENDOR CPU_CONFIG)
    if(NOT FB_BUILD_OPENBLAS_FROM_SOURCE)
        return()
    endif()
    
    message(STATUS "Configuring OpenBLAS build from source (${CPU_CONFIG} optimized)")
    
    set(OPENBLAS_SOURCE_DIR "${CMAKE_BINARY_DIR}/openblas-src")
    set(OPENBLAS_BINARY_DIR "${CMAKE_BINARY_DIR}/openblas-build")
    set(OPENBLAS_INSTALL_DIR "${FB_BACKENDS_INSTALL_PREFIX}/openblas")
    
    # OpenBLAS target architecture
    if(CPU_VENDOR STREQUAL "AMD")
        if(CPU_CONFIG STREQUAL "znver4")
            set(OPENBLAS_TARGET "ZEN")  # OpenBLAS doesn't have separate Zen4 yet
        elseif(CPU_CONFIG STREQUAL "znver3")
            set(OPENBLAS_TARGET "ZEN")
        else()
            set(OPENBLAS_TARGET "ZEN")
        endif()
    elseif(CPU_VENDOR STREQUAL "Intel")
        if(CPU_CONFIG MATCHES "skylake|cascadelake|icelake|sapphirerapids")
            set(OPENBLAS_TARGET "SKYLAKEX")
        else()
            set(OPENBLAS_TARGET "HASWELL")
        endif()
    else()
        set(OPENBLAS_TARGET "GENERIC")
    endif()
    
    # Build flags
    set(OPENBLAS_MAKE_FLAGS
        "TARGET=${OPENBLAS_TARGET}"
        "USE_OPENMP=1"
        "USE_THREAD=1"
        "NUM_THREADS=128"
        "DYNAMIC_ARCH=1"  # Support runtime detection
        "NO_LAPACKE=0"
        "BUILD_LAPACK_DEPRECATED=0"
    )
    
    # Platform-specific build configuration
    if(WIN32)
        # Prefer precompiled OpenBLAS binaries on Windows for reliability.
        if(FB_USE_PREBUILT_OPENBLAS_WINDOWS)
            set(_openblas_prebuilt_root "${FB_OPENBLAS_WINDOWS_PREBUILT_ROOT}")

            if(NOT _openblas_prebuilt_root)
                set(_openblas_prebuilt_candidates
                    "C:/libraries/OpenBLAS-0.3.30-x64"
                    "C:/libraries/OpenBLAS-0.3.29-x64"
                    "C:/libraries/OpenBLAS-0.3.28-x64"
                    "C:/Program Files/OpenBLAS"
                    "C:/OpenBLAS"
                )

                foreach(_cand IN LISTS _openblas_prebuilt_candidates)
                    if(EXISTS "${_cand}/include" AND (EXISTS "${_cand}/lib/openblas.lib" OR EXISTS "${_cand}/lib/libopenblas.a"))
                        set(_openblas_prebuilt_root "${_cand}")
                        break()
                    endif()
                endforeach()
            endif()

            if(_openblas_prebuilt_root)
                message(STATUS "Using precompiled OpenBLAS on Windows: ${_openblas_prebuilt_root}")

                if(EXISTS "${_openblas_prebuilt_root}/lib/openblas.lib")
                    set(_openblas_lib_file "openblas.lib")
                elseif(EXISTS "${_openblas_prebuilt_root}/lib/libopenblas.a")
                    set(_openblas_lib_file "libopenblas.a")
                else()
                    set(_openblas_lib_file "openblas.lib")
                endif()

                ExternalProject_Add(openblas-backend
                    DOWNLOAD_COMMAND ""
                    UPDATE_COMMAND ""
                    PATCH_COMMAND ""
                    SOURCE_DIR ${_openblas_prebuilt_root}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND
                        ${CMAKE_COMMAND} -E make_directory ${OPENBLAS_INSTALL_DIR}
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_openblas_prebuilt_root}/include ${OPENBLAS_INSTALL_DIR}/include
                        COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENBLAS_INSTALL_DIR}/lib
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_openblas_prebuilt_root}/lib ${OPENBLAS_INSTALL_DIR}/lib
                    LOG_INSTALL TRUE
                )

                set(OpenBLAS_FOUND TRUE PARENT_SCOPE)
                set(OpenBLAS_INCLUDE_DIRS "${OPENBLAS_INSTALL_DIR}/include" PARENT_SCOPE)
                set(OpenBLAS_LIBRARIES "${OPENBLAS_INSTALL_DIR}/lib/${_openblas_lib_file}" PARENT_SCOPE)
                message(STATUS "OpenBLAS will be staged to: ${OPENBLAS_INSTALL_DIR}")
                return()
            else()
                if(NOT DEFINED FB_OPENBLAS_WINDOWS_PREBUILT_VERSION OR FB_OPENBLAS_WINDOWS_PREBUILT_VERSION STREQUAL "")
                    set(FB_OPENBLAS_WINDOWS_PREBUILT_VERSION "0.3.30")
                endif()

                if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                    set(_openblas_pkg_arch "x64")
                else()
                    set(_openblas_pkg_arch "x86")
                endif()

                set(_openblas_pkg_url
                    "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${FB_OPENBLAS_WINDOWS_PREBUILT_VERSION}/OpenBLAS-${FB_OPENBLAS_WINDOWS_PREBUILT_VERSION}-${_openblas_pkg_arch}.zip")

                message(STATUS "Downloading precompiled OpenBLAS on Windows: ${_openblas_pkg_url}")

                ExternalProject_Add(openblas-backend
                    URL ${_openblas_pkg_url}
                    SOURCE_DIR ${OPENBLAS_SOURCE_DIR}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND
                        ${CMAKE_COMMAND} -E make_directory ${OPENBLAS_INSTALL_DIR}
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPENBLAS_SOURCE_DIR}/include ${OPENBLAS_INSTALL_DIR}/include
                        COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENBLAS_INSTALL_DIR}/lib
                        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPENBLAS_SOURCE_DIR}/lib ${OPENBLAS_INSTALL_DIR}/lib
                    LOG_DOWNLOAD TRUE
                    LOG_INSTALL TRUE
                )

                set(OpenBLAS_FOUND TRUE PARENT_SCOPE)
                set(OpenBLAS_INCLUDE_DIRS "${OPENBLAS_INSTALL_DIR}/include" PARENT_SCOPE)
                set(OpenBLAS_LIBRARIES "${OPENBLAS_INSTALL_DIR}/lib/openblas.lib" PARENT_SCOPE)
                message(STATUS "OpenBLAS will be staged to: ${OPENBLAS_INSTALL_DIR}")
                return()
            endif()
        endif()

        # Native Windows build using external PowerShell script
        # This avoids CMake generator inheritance conflicts
        # The script builds OpenBLAS independently with clang-cl (C99/VLA + MSVC ABI + OpenMP 5.0+)
        set(BUILD_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/build_openblas_msvc_conly.ps1")
        set(BUILD_WRAPPER_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/build_openblas_external_wrapper.ps1")
        
        # Download source first
        ExternalProject_Add(openblas-backend
            GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
            GIT_TAG v0.3.30
            GIT_SHALLOW TRUE
            SOURCE_DIR ${OPENBLAS_SOURCE_DIR}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File "${BUILD_WRAPPER_SCRIPT}"
                -BuildScript "${BUILD_SCRIPT}"
                -SourceDir "${OPENBLAS_SOURCE_DIR}"
                -InstallDir "${OPENBLAS_INSTALL_DIR}"
                -Target "${OPENBLAS_TARGET}"
                -NumCores ${NUM_CORES}
            INSTALL_COMMAND ""
            LOG_DOWNLOAD TRUE
            LOG_BUILD TRUE
        )
    else()
        # Unix-like systems: use make-based build
        list(APPEND OPENBLAS_MAKE_FLAGS "PREFIX=${OPENBLAS_INSTALL_DIR}")
        
        ExternalProject_Add(openblas-backend
            GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
            GIT_TAG v0.3.30
            GIT_SHALLOW TRUE
            SOURCE_DIR ${OPENBLAS_SOURCE_DIR}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND make ${OPENBLAS_MAKE_FLAGS} -j${NUM_CORES}
            INSTALL_COMMAND make install
            BUILD_IN_SOURCE TRUE
            LOG_DOWNLOAD TRUE
            LOG_BUILD TRUE
            LOG_INSTALL TRUE
        )
    endif()
    
    # Export variables
    set(OpenBLAS_FOUND TRUE PARENT_SCOPE)
    set(OpenBLAS_INCLUDE_DIRS "${OPENBLAS_INSTALL_DIR}/include" PARENT_SCOPE)
    set(OpenBLAS_LIBRARIES "${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a" PARENT_SCOPE)
    
    message(STATUS "OpenBLAS will be built and installed to: ${OPENBLAS_INSTALL_DIR}")
endfunction()

#==============================================================================
# BUILD CLBLAST FROM SOURCE
#==============================================================================

function(build_clblast_from_source HAS_OPENCL)
    if(NOT FB_BUILD_CLBLAST_FROM_SOURCE OR NOT HAS_OPENCL)
        return()
    endif()
    
    message(STATUS "Configuring CLBlast build from source")
    
    set(CLBLAST_SOURCE_DIR "${CMAKE_BINARY_DIR}/clblast-src")
    set(CLBLAST_BINARY_DIR "${CMAKE_BINARY_DIR}/clblast-build")
    set(CLBLAST_INSTALL_DIR "${FB_BACKENDS_INSTALL_PREFIX}/clblast")
    
    ExternalProject_Add(clblast-backend
        GIT_REPOSITORY https://github.com/CNugteren/CLBlast.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        SOURCE_DIR ${CLBLAST_SOURCE_DIR}
        BINARY_DIR ${CLBLAST_BINARY_DIR}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CLBLAST_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=ON
            -DTUNERS=OFF
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CXX_STANDARD=17
            -DCMAKE_CXX_STANDARD_REQUIRED=ON
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release -j${NUM_CORES}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config Release --prefix ${CLBLAST_INSTALL_DIR}
        LOG_DOWNLOAD TRUE
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
    )
    
    # Export variables
    set(CLBlast_FOUND TRUE PARENT_SCOPE)
    set(CLBlast_INCLUDE_DIRS "${CLBLAST_INSTALL_DIR}/include" PARENT_SCOPE)
    set(CLBlast_LIBRARIES "${CLBLAST_INSTALL_DIR}/lib/libclblast.so" PARENT_SCOPE)
    
    message(STATUS "CLBlast will be built and installed to: ${CLBLAST_INSTALL_DIR}")
endfunction()

#==============================================================================
# MAIN CONFIGURATION
#==============================================================================

#==============================================================================
# OXIBLAS FROM SOURCE (cargo build)
#==============================================================================

function(build_oxiblas_from_source)
    if(NOT FB_BUILD_OXIBLAS_FROM_SOURCE)
        return()
    endif()

    # Locate cargo
    find_program(CARGO_EXECUTABLE cargo)
    if(NOT CARGO_EXECUTABLE)
        message(WARNING "OxiBLAS from-source build requested but 'cargo' not found. "
                        "Install Rust 1.85+ from https://rustup.rs/ and try again.")
        return()
    endif()

    # Determine source directory
    if(OXIBLAS_CARGO_SOURCE_DIR AND EXISTS "${OXIBLAS_CARGO_SOURCE_DIR}/Cargo.toml")
        set(OXIBLAS_SRC "${OXIBLAS_CARGO_SOURCE_DIR}")
        set(_oxiblas_user_source TRUE)
    else()
        # Clone from GitHub if not provided
        set(OXIBLAS_SRC "${CMAKE_BINARY_DIR}/oxiblas-src")
        set(_oxiblas_user_source FALSE)
        if(EXISTS "${OXIBLAS_SRC}/Cargo.toml")
            set(_oxiblas_refresh_checkout FALSE)
            if(NOT EXISTS "${OXIBLAS_SRC}/crates/oxiblas-ffi/Cargo.toml")
                set(_oxiblas_refresh_checkout TRUE)
            else()
                file(READ "${OXIBLAS_SRC}/Cargo.toml" _oxiblas_existing_manifest)
                if(NOT _oxiblas_existing_manifest MATCHES "\"crates/oxiblas-ffi\"")
                    set(_oxiblas_refresh_checkout TRUE)
                endif()
            endif()

            if(_oxiblas_refresh_checkout)
                message(STATUS
                    "Refreshing OxiBLAS checkout to ${FB_OXIBLAS_FFI_GIT_TAG} because the existing clone no longer exposes oxiblas-ffi as a workspace member")
                file(REMOVE_RECURSE "${OXIBLAS_SRC}")
            endif()
        endif()
        if(NOT EXISTS "${OXIBLAS_SRC}/Cargo.toml")
            message(STATUS "Cloning OxiBLAS repository (${FB_OXIBLAS_FFI_GIT_TAG})...")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch "${FB_OXIBLAS_FFI_GIT_TAG}" --single-branch
                    https://github.com/cool-japan/oxiblas.git
                    "${OXIBLAS_SRC}"
                RESULT_VARIABLE _clone_result
            )
            if(NOT _clone_result EQUAL 0)
                message(WARNING "Failed to clone OxiBLAS repository. "
                                "Set OXIBLAS_CARGO_SOURCE_DIR to an existing checkout.")
                return()
            endif()
        endif()
    endif()

    if(NOT EXISTS "${OXIBLAS_SRC}/crates/oxiblas-ffi/Cargo.toml")
        message(WARNING
            "The selected OxiBLAS checkout does not contain crates/oxiblas-ffi/Cargo.toml. "
            "Upstream retired the C FFI in v0.2.0+. "
            "Set FB_OXIBLAS_FFI_GIT_TAG to a pre-v0.2.0 tag such as v0.1.1, "
            "point OXIBLAS_CARGO_SOURCE_DIR at a compatible checkout, "
            "or disable FB_BUILD_OXIBLAS_FROM_SOURCE.")
        return()
    endif()

    file(READ "${OXIBLAS_SRC}/Cargo.toml" _oxiblas_workspace_manifest)
    if(NOT _oxiblas_workspace_manifest MATCHES "\"crates/oxiblas-ffi\"")
        if(_oxiblas_user_source)
            message(WARNING
                "The selected OxiBLAS checkout has retired oxiblas-ffi from workspace members, so "
                "'cargo build -p oxiblas-ffi' cannot succeed. "
                "Point OXIBLAS_CARGO_SOURCE_DIR at a pre-v0.2.0 checkout, "
                "set FB_OXIBLAS_FFI_GIT_TAG to a compatible tag such as v0.1.1 for auto-clone mode, "
                "or disable FB_BUILD_OXIBLAS_FROM_SOURCE.")
        else()
            message(WARNING
                "The auto-cloned OxiBLAS checkout retired oxiblas-ffi from workspace members unexpectedly. "
                "Check FB_OXIBLAS_FFI_GIT_TAG (currently ${FB_OXIBLAS_FFI_GIT_TAG}) or disable FB_BUILD_OXIBLAS_FROM_SOURCE.")
        endif()
        return()
    endif()

    set(OXIBLAS_INSTALL_DIR "${FB_BACKENDS_INSTALL_PREFIX}/oxiblas")
    set(OXIBLAS_TARGET_DIR  "${CMAKE_BINARY_DIR}/oxiblas-cargo-target")

    message(STATUS "Building OxiBLAS FFI from source:")
    message(STATUS "  Source : ${OXIBLAS_SRC}")
    message(STATUS "  Output : ${OXIBLAS_INSTALL_DIR}/lib")
    message(STATUS "  cargo  : ${CARGO_EXECUTABLE}")

    # Run cargo build
    execute_process(
        COMMAND ${CARGO_EXECUTABLE} build --release -p oxiblas-ffi
            --target-dir "${OXIBLAS_TARGET_DIR}"
        WORKING_DIRECTORY "${OXIBLAS_SRC}"
        RESULT_VARIABLE _cargo_result
    )
    if(NOT _cargo_result EQUAL 0)
        message(WARNING "cargo build --release -p oxiblas-ffi failed "
                        "(exit ${_cargo_result}). OxiBLAS will not be available.")
        return()
    endif()

    # Install the built artifacts
    file(MAKE_DIRECTORY "${OXIBLAS_INSTALL_DIR}/lib")
    foreach(_lib
            "${OXIBLAS_TARGET_DIR}/release/${CMAKE_SHARED_LIBRARY_PREFIX}oxiblas_ffi${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "${OXIBLAS_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}oxiblas_ffi${CMAKE_STATIC_LIBRARY_SUFFIX}")
        if(EXISTS "${_lib}")
            file(COPY "${_lib}" DESTINATION "${OXIBLAS_INSTALL_DIR}/lib")
        endif()
    endforeach()

    message(STATUS "✓ OxiBLAS FFI installed to ${OXIBLAS_INSTALL_DIR}/lib")
endfunction()

if(FB_BUILD_BACKENDS_FROM_SOURCE)
    # Disable BackendInstaller's conflicting OpenBLAS build
    set(BUILD_OPENBLAS_FROM_SOURCE OFF CACHE BOOL "Disabled by FB_BUILD_BACKENDS_FROM_SOURCE" FORCE)
    
    message(STATUS "")
    message(STATUS "=================================================================")
    message(STATUS "  Building Backends From Source for Maximum Performance")
    message(STATUS "=================================================================")
    
    # Detect hardware
    detect_cpu_architecture(CPU_VENDOR CPU_ARCH CPU_CONFIG)
    detect_gpu_architectures(HAS_NVIDIA CUDA_ARCH HAS_AMD ROCM_ARCH HAS_OPENCL)
    
    message(STATUS "")
    message(STATUS "Hardware Configuration:")
    message(STATUS "  CPU Vendor: ${CPU_VENDOR}")
    message(STATUS "  CPU Architecture: ${CPU_ARCH}")
    message(STATUS "  CPU Config: ${CPU_CONFIG}")
    if(HAS_NVIDIA)
        message(STATUS "  NVIDIA GPU: Yes (CUDA ${FB_DETECTED_CUDA_VERSION})")
    endif()
    if(HAS_AMD)
        message(STATUS "  AMD GPU: Yes (ROCm)")
    endif()
    if(HAS_OPENCL)
        message(STATUS "  OpenCL: Yes")
    endif()
    message(STATUS "")
    
    # Get optimized compiler flags
    get_optimized_compiler_flags(${CPU_VENDOR} ${CPU_CONFIG} OPT_C_FLAGS OPT_CXX_FLAGS)
    message(STATUS "Optimized C Flags: ${OPT_C_FLAGS}")
    message(STATUS "Optimized CXX Flags: ${OPT_CXX_FLAGS}")
    message(STATUS "")
    
    # Build backends
    message(STATUS "Building backends:")
    build_blis_from_source(${CPU_ARCH})
    build_openblas_from_source(${CPU_VENDOR} ${CPU_CONFIG})
    build_clblast_from_source(${HAS_OPENCL})
    build_oxiblas_from_source()
    
    message(STATUS "")
    message(STATUS "All backends will be installed to: ${FB_BACKENDS_INSTALL_PREFIX}")
    message(STATUS "=================================================================")
    message(STATUS "")
    
    # Add install directory to CMAKE_PREFIX_PATH so later find_package calls can see built backends.
    if(NOT FB_BACKENDS_INSTALL_PREFIX IN_LIST CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${FB_BACKENDS_INSTALL_PREFIX}")
    endif()
endif()
