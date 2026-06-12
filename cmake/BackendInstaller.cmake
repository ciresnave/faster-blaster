# BackendInstaller.cmake
# Automatic detection and optional installation of BLAS/LAPACK backends

include(ExternalProject)

# Options
option(AUTO_DETECT_BACKENDS "Automatically detect available backends" ON)
option(AUTO_INSTALL_BACKENDS "Prompt to install missing backends" ON)
option(BUILD_OPENBLAS_FROM_SOURCE "Build OpenBLAS from source with full features" ON)
option(INTERACTIVE_BACKEND_SETUP "Prompt for backend installation decisions" ON)

set(FB_CAN_PROMPT FALSE)
if(INTERACTIVE_BACKEND_SETUP AND CMAKE_INTERACTIVE)
    set(FB_CAN_PROMPT TRUE)
endif()

# Detect CPU vendor.
# Uses cmake_host_system_information — a built-in zero-subprocess CMake query.
# No PowerShell, no WMI, no wmic, no subprocess hang risk.  Runs instantly on
# every reconfigure and always reflects the actual current machine.
function(detect_cpu_vendor OUT_VENDOR OUT_MODEL)
    # PROCESSOR_DESCRIPTION (CMake 3.10+) gives the full marketing name string,
    # e.g. "Intel(R) Core(TM) i7-12700K" or "AMD Ryzen 9 7950X".
    cmake_host_system_information(RESULT _cpu_info QUERY PROCESSOR_DESCRIPTION)
    if(NOT _cpu_info)
        # Fallback for older CMake or unusual platforms.
        cmake_host_system_information(RESULT _cpu_info QUERY PROCESSOR_NAME)
    endif()
    if(NOT _cpu_info)
        set(_cpu_info "${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    string(STRIP "${_cpu_info}" _cpu_info)

    if(_cpu_info MATCHES "AMD|Ryzen|Threadripper|EPYC")
        set(_vendor "AMD")
    elseif(_cpu_info MATCHES "Intel|Core|Xeon|Pentium|Celeron")
        set(_vendor "Intel")
    else()
        set(_vendor "Unknown")
    endif()

    set(${OUT_VENDOR} "${_vendor}"  PARENT_SCOPE)
    set(${OUT_MODEL}  "${_cpu_info}" PARENT_SCOPE)
endfunction()

# Detect available GPUs
function(detect_gpus OUT_HAS_NVIDIA OUT_HAS_AMD OUT_HAS_OPENCL)
    set(${OUT_HAS_NVIDIA} FALSE PARENT_SCOPE)
    set(${OUT_HAS_AMD} FALSE PARENT_SCOPE)
    set(${OUT_HAS_OPENCL} FALSE PARENT_SCOPE)
    
    # Check for NVIDIA CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(${OUT_HAS_NVIDIA} TRUE PARENT_SCOPE)
        message(STATUS "  NVIDIA CUDA Toolkit: ${CUDAToolkit_VERSION}")
    endif()
    
    # Check for AMD ROCm.
    # Guard: only call find_package(hip) if a ROCm directory actually exists on
    # disk. Without this guard, cmake searches the Windows registry for
    # hipConfig.cmake and — if a stale/partial ROCm install left registry entries —
    # loads a broken package config that can block cmake for minutes while trying
    # to run hipcc or enumerate GPU devices.
    set(_hip_candidate_paths
        "$ENV{ROCM_PATH}"
        "$ENV{HIP_PATH}"
        "C:/Program Files/AMD/ROCm"
        "/opt/rocm"
    )
    if(WIN32)
        file(GLOB _hip_versioned_dirs LIST_DIRECTORIES TRUE "C:/Program Files/AMD/ROCm/*")
    else()
        file(GLOB _hip_versioned_dirs LIST_DIRECTORIES TRUE "/opt/rocm-*")
    endif()
    list(APPEND _hip_candidate_paths ${_hip_versioned_dirs})
    set(_hip_present FALSE)
    foreach(_p IN LISTS _hip_candidate_paths)
        if(_p STREQUAL "")
            continue()
        endif()
        if(EXISTS "${_p}/bin/hipcc" OR EXISTS "${_p}/bin/hipcc.exe" OR EXISTS "${_p}/bin/hipcc.bat"
           OR EXISTS "${_p}/include/hip/hip_runtime.h")
            set(_hip_present TRUE)
            break()
        endif()
    endforeach()
    unset(_hip_candidate_paths)
    unset(_hip_versioned_dirs)

    if(_hip_present)
        set(${OUT_HAS_AMD} TRUE PARENT_SCOPE)
        message(STATUS "  AMD ROCm/HIP: Present")
    else()
        message(STATUS "  AMD ROCm/HIP: Not present")
    endif()
    unset(_hip_present)
    
    # Check for OpenCL (catches all GPUs including AMD integrated)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(${OUT_HAS_OPENCL} TRUE PARENT_SCOPE)
        message(STATUS "  OpenCL: ${OpenCL_VERSION_STRING}")
        message(STATUS "    → Can detect NVIDIA, AMD (integrated/discrete), Intel, and other GPUs")
    endif()
endfunction()

# Check for available CPU backends
function(check_cpu_backends OUT_REPORT)
    set(REPORT "")
    
    # OpenBLAS
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
        string(APPEND REPORT "  ✓ OpenBLAS: ${OpenBLAS_VERSION} (found)\n")
    else()
        string(APPEND REPORT "  ✗ OpenBLAS: Not found\n")
    endif()
    
    # Intel MKL
    find_package(MKL QUIET)
    if(MKL_FOUND)
        string(APPEND REPORT "  ✓ Intel MKL: (found)\n")
    else()
        string(APPEND REPORT "  ✗ Intel MKL: Not found\n")
    endif()
    
    # AOCL (AMD - provides BLIS for BLAS and libFLAME for LAPACK)
    find_package(AOCL QUIET)
    if(AOCL_FOUND)
        if(AOCL_FLAME_LIBRARY)
            string(APPEND REPORT "  ✓ AMD AOCL: BLIS (BLAS) + libFLAME (LAPACK) found\n")
        else()
            string(APPEND REPORT "  ✓ AMD AOCL: BLIS (BLAS only) - libFLAME not found\n")
        endif()
    else()
        string(APPEND REPORT "  ✗ AMD AOCL: Not found\n")
    endif()
    
    set(${OUT_REPORT} "${REPORT}" PARENT_SCOPE)
endfunction()

# Check for available GPU backends
function(check_gpu_backends OUT_REPORT)
    set(REPORT "")
    
    # cuBLAS (comes with CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        string(APPEND REPORT "  ✓ cuBLAS: ${CUDAToolkit_VERSION} (found)\n")
    else()
        string(APPEND REPORT "  ✗ cuBLAS: Not found\n")
    endif()
    
    # rocBLAS (AMD GPU - includes rocBLAS, rocSOLVER, hipBLAS)
    find_package(ROCBLAS QUIET)
    if(ROCBLAS_FOUND)
        string(APPEND REPORT "  ✓ AMD rocBLAS + rocSOLVER + hipBLAS: Found\n")
    else()
        string(APPEND REPORT "  ✗ AMD rocBLAS: Not found\n")
    endif()
    
    # CLBlast
    find_package(CLBlast QUIET)
    if(CLBlast_FOUND)
        string(APPEND REPORT "  ✓ CLBlast: (found)\n")
    else()
        string(APPEND REPORT "  ✗ CLBlast: Not found\n")
    endif()
    
    set(${OUT_REPORT} "${REPORT}" PARENT_SCOPE)
endfunction()

# Build OpenBLAS from source with full threading support
function(build_openblas_from_source)
    message(STATUS "Building OpenBLAS from source with threading support...")
    
    set(OPENBLAS_VERSION "0.3.30")
    set(OPENBLAS_URL "https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz")
    
    ExternalProject_Add(
        openblas_external
        URL ${OPENBLAS_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX ${CMAKE_BINARY_DIR}/external/openblas
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external/install
            -DBUILD_SHARED_LIBS=ON
            -DUSE_THREAD=1
            -DUSE_OPENMP=1
            -DDYNAMIC_ARCH=ON
            -DBUILD_TESTING=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config ${CMAKE_BUILD_TYPE}
    )
    
    # Set variables for find_package to locate our custom build
    set(OpenBLAS_DIR ${CMAKE_BINARY_DIR}/external/install/lib/cmake/openblas PARENT_SCOPE)
    set(OPENBLAS_BUILT_FROM_SOURCE TRUE PARENT_SCOPE)
    
    message(STATUS "OpenBLAS will be built with threading support enabled")
    message(STATUS "  Build will occur during 'cmake --build'")
endfunction()

# Interactive prompt helper
function(prompt_user PROMPT DEFAULT_VALUE OUT_VAR)
    if(NOT FB_CAN_PROMPT OR DEFINED ${OUT_VAR})
        set(${OUT_VAR} ${DEFAULT_VALUE} PARENT_SCOPE)
        return()
    endif()

    message("")
    message("${PROMPT}")
    if(DEFAULT_VALUE)
        message("  [Y/n] (default: yes): ")
    else()
        message("  [y/N] (default: no): ")
    endif()

    # CMake interactive input is limited; keep defaults and let users override with -D flags.
    set(${OUT_VAR} ${DEFAULT_VALUE} PARENT_SCOPE)
endfunction()

# Install Intel MKL
function(install_intel_mkl)
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS " Intel MKL Installation")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "Intel MKL provides optimized BLAS/LAPACK for Intel CPUs")
    message(STATUS "")
    message(STATUS "Download options:")
    message(STATUS "  1. Intel oneAPI Math Kernel Library (free)")
    message(STATUS "     https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html")
    message(STATUS "  2. Via vcpkg: vcpkg install intel-mkl")
    message(STATUS "")
    message(STATUS "After installation, reconfigure with: cmake ..")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
endfunction()

# Install AMD AOCL (with auto-download)
function(install_amd_aocl)
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS " AMD AOCL Installation")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "AMD AOCL provides optimized BLIS/FLAME for AMD CPUs")
    message(STATUS "  → 2-4x faster than OpenBLAS on AMD Ryzen/EPYC")
    message(STATUS "")
    
    if(WIN32)
        message(STATUS "Windows Installation:")
        message(STATUS "  1. Download installer from:")
        message(STATUS "     https://www.amd.com/en/developer/aocl/blis.html")
        message(STATUS "  2. Run AOCL-Windows installer")
        message(STATUS "  3. Default install: C:/Program Files/AMD/AOCL-Windows")
        message(STATUS "  4. Reconfigure: cmake ..")
    else()
        message(STATUS "Linux Installation:")
        message(STATUS "  wget https://www.amd.com/en/developer/aocl/blis/eula/blis-download.html")
        message(STATUS "  tar xzf aocl-*.tar.gz")
        message(STATUS "  sudo mv aocl-linux /opt/AMD/aocl")
        message(STATUS "  cmake ..")
    endif()
    
    message(STATUS "")
    message(STATUS "After installation, AOCL will be auto-detected")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
endfunction()

# Install CUDA Toolkit (show instructions)
function(install_cuda_toolkit)
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS " CUDA Toolkit Installation")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
    message(STATUS "CUDA Toolkit provides fastest GPU acceleration for NVIDIA GPUs")
    message(STATUS "")
    message(STATUS "Windows:")
    message(STATUS "  Download: https://developer.nvidia.com/cuda-downloads")
    message(STATUS "  Select: Windows → x86_64 → 11/10 → exe (local)")
    message(STATUS "  Default install path: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x")
    message(STATUS "")
    message(STATUS "Linux:")
    message(STATUS "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit")
    message(STATUS "  Or download: https://developer.nvidia.com/cuda-downloads")
    message(STATUS "")
    message(STATUS "After installation:")
    message(STATUS "  1. Restart your terminal")
    message(STATUS "  2. Reconfigure: cmake ..")
    message(STATUS "  3. cuBLAS will be automatically detected and enabled")
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
endfunction()

# Install AMD ROCm (show instructions)
function(install_amd_rocm)
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS " AMD ROCm Installation")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
    message(STATUS "ROCm provides fastest GPU acceleration for AMD discrete GPUs")
    message(STATUS "Note: ROCm does NOT support AMD integrated GPUs (like Radeon 610M)")
    message(STATUS "      Use CLBlast for integrated GPUs")
    message(STATUS "")
    message(STATUS "Windows:")
    message(STATUS "  Download: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html")
    message(STATUS "  Install HIP SDK for Windows")
    message(STATUS "")
    message(STATUS "Linux (Ubuntu 22.04/24.04):")
    message(STATUS "  wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_5.7.50700-1_all.deb")
    message(STATUS "  sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb")
    message(STATUS "  sudo amdgpu-install --usecase=rocm")
    message(STATUS "")
    message(STATUS "After installation:")
    message(STATUS "  1. Restart your terminal")
    message(STATUS "  2. Reconfigure: cmake ..")
    message(STATUS "  3. rocBLAS will be automatically detected and enabled")
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
endfunction()

# Install CLBlast (via vcpkg auto-install)
function(install_clblast)
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS " CLBlast Installation")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "CLBlast provides GPU acceleration via OpenCL")
    message(STATUS "Works with: NVIDIA, AMD (all GPUs), Intel, Apple, etc.")
    message(STATUS "")
    
    if(DEFINED ENV{VCPKG_ROOT})
        message(STATUS "Option 1: Auto-install via vcpkg (RECOMMENDED)")
        message(STATUS "  vcpkg detected at: $ENV{VCPKG_ROOT}")
        message(STATUS "")
        message(STATUS "Option 2: Build from source")
        message(STATUS "  https://github.com/CNugteren/CLBlast")
        message(STATUS "")
        
        message(STATUS "Auto-install CLBlast via vcpkg? [Y/n]")
        set(AUTO_INSTALL_CLBLAST ON)  # Default YES
        
        if(AUTO_INSTALL_CLBLAST)
            message(STATUS "→ Installing CLBlast via vcpkg...")
            execute_process(
                COMMAND "$ENV{VCPKG_ROOT}/vcpkg" install clblast opencl
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE VCPKG_RESULT
            )
            
            if(VCPKG_RESULT EQUAL 0)
                message(STATUS "  ✅ CLBlast installed successfully")
                message(STATUS "  Please reconfigure: cmake ..")
            else()
                message(STATUS "  ⚠️ vcpkg install failed, falling back to manual")
                message(STATUS "  Run manually: vcpkg install clblast opencl")
            endif()
        endif()
    else()
        message(STATUS "vcpkg not detected. Install options:")
        message(STATUS "  1. Install vcpkg, then: vcpkg install clblast")
        message(STATUS "  2. Build from source: https://github.com/CNugteren/CLBlast")
        message(STATUS "")
        message(STATUS "After installation, reconfigure with: cmake ..")
    endif()
    
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
endfunction()

# Main detection and reporting
if(AUTO_DETECT_BACKENDS)
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS " Backend Detection & Configuration")
    message(STATUS "══════════════════════════════════════════════════════")
    
    # Detect hardware
    detect_cpu_vendor(CPU_VENDOR CPU_MODEL)
    message(STATUS "CPU: ${CPU_MODEL}")
    message(STATUS "CPU Vendor: ${CPU_VENDOR}")
    
    detect_gpus(HAS_NVIDIA HAS_AMD_ROCM HAS_OPENCL)
    
    # Check available backends
    message(STATUS "")
    message(STATUS "CPU Backends:")
    check_cpu_backends(CPU_BACKEND_REPORT)
    message(STATUS "${CPU_BACKEND_REPORT}")
    
    message(STATUS "GPU Backends:")
    check_gpu_backends(GPU_BACKEND_REPORT)
    message(STATUS "${GPU_BACKEND_REPORT}")
    
    message(STATUS "══════════════════════════════════════════════════════")
    
    # Interactive installation prompts
    set(MISSING_BACKENDS FALSE)
    
    # Check for missing CPU backends based on hardware
    if(CPU_VENDOR STREQUAL "AMD" AND NOT BLIS_LIB)
        set(MISSING_BACKENDS TRUE)
        if(FB_CAN_PROMPT)
            message(STATUS "")
            message(STATUS "💡 AMD CPU detected without AOCL-BLIS")
            message(STATUS "   AOCL provides 2-4x better performance on AMD CPUs")
            prompt_user("   Install AMD AOCL? [Y/n]" TRUE INSTALL_AOCL_PROMPT)
            if(INSTALL_AOCL_PROMPT)
                install_amd_aocl()
            endif()
        endif()
    endif()
    
    if(CPU_VENDOR STREQUAL "Intel" AND NOT MKL_FOUND)
        set(MISSING_BACKENDS TRUE)
        if(FB_CAN_PROMPT)
            message(STATUS "")
            message(STATUS "💡 Intel CPU detected without Intel MKL")
            message(STATUS "   MKL provides 2-4x better performance on Intel CPUs")
            prompt_user("   Install Intel MKL? [Y/n]" TRUE INSTALL_MKL_PROMPT)
            if(INSTALL_MKL_PROMPT)
                install_intel_mkl()
            endif()
        endif()
    endif()
    
    # OpenBLAS threading check
    if(OpenBLAS_FOUND AND NOT OPENBLAS_BUILT_FROM_SOURCE)
        if(FB_CAN_PROMPT AND BUILD_OPENBLAS_FROM_SOURCE)
            message(STATUS "")
            message(STATUS "💡 OpenBLAS found (likely from vcpkg)")
            message(STATUS "   vcpkg's OpenBLAS is single-threaded (slower)")
            message(STATUS "   We can build OpenBLAS from source with full threading support")
            message(STATUS "   This will enable multi-threaded CPU acceleration")
            # Automatically enable if the option is ON
            if(BUILD_OPENBLAS_FROM_SOURCE)
                message(STATUS "   → Building threaded OpenBLAS from source (BUILD_OPENBLAS_FROM_SOURCE=ON)")
                build_openblas_from_source()
            endif()
        endif()
    endif()
    
    # Check for missing GPU backends
    # Re-check CUDAToolkit in case it wasn't found earlier
    if(HAS_NVIDIA)
        find_package(CUDAToolkit QUIET)
        if(NOT CUDAToolkit_FOUND)
            set(MISSING_BACKENDS TRUE)
            if(FB_CAN_PROMPT)
                message(STATUS "")
                message(STATUS "💡 NVIDIA GPU detected without CUDA Toolkit")
                message(STATUS "   CUDA provides fastest performance for NVIDIA GPUs")
                prompt_user("   Show CUDA installation instructions? [Y/n]" TRUE SHOW_CUDA_INSTALL)
                if(SHOW_CUDA_INSTALL)
                    install_cuda_toolkit()
                endif()
            endif()
        else()
            message(STATUS "")
            message(STATUS "✅ NVIDIA GPU with CUDA Toolkit ${CUDAToolkit_VERSION}")
        endif()
    endif()
    
    if(HAS_AMD_ROCM AND NOT rocblas_FOUND)
        set(MISSING_BACKENDS TRUE)
        if(FB_CAN_PROMPT)
            message(STATUS "")
            message(STATUS "💡 AMD discrete GPU detected without ROCm")
            message(STATUS "   ROCm provides fastest performance for AMD discrete GPUs")
            prompt_user("   Show ROCm installation instructions? [Y/n]" TRUE SHOW_ROCM_INSTALL)
            if(SHOW_ROCM_INSTALL)
                install_amd_rocm()
            endif()
        endif()
    endif()
    
    # OpenCL backend recommendation (for ALL GPUs including AMD integrated)
    if(HAS_OPENCL)
        find_package(CLBlast QUIET)
        if(NOT CLBlast_FOUND)
            set(MISSING_BACKENDS TRUE)
            if(FB_CAN_PROMPT)
                message(STATUS "")
                message(STATUS "💡 OpenCL GPUs detected without CLBlast")
                message(STATUS "   CLBlast enables GPU acceleration via OpenCL")
                message(STATUS "   Works with: NVIDIA, AMD (integrated & discrete), Intel, etc.")
                message(STATUS "   ℹ️  Especially useful for AMD integrated GPUs (like Radeon 610M)")
                prompt_user("   Install CLBlast? [Y/n]" TRUE INSTALL_CLBLAST_PROMPT)
                if(INSTALL_CLBLAST_PROMPT)
                    install_clblast()
                endif()
            endif()
        else()
            message(STATUS "")
            message(STATUS "✅ OpenCL available with CLBlast")
            message(STATUS "   → All GPUs can be accelerated (NVIDIA, AMD, Intel, etc.)")
        endif()
    endif()
    
    # Summary
    message(STATUS "")
    message(STATUS "══════════════════════════════════════════════════════")
    if(NOT MISSING_BACKENDS)
        message(STATUS " ✅ All recommended backends installed!")
    else()
        message(STATUS " ℹ️  Some recommended backends are missing")
        message(STATUS "   You can install them later and reconfigure")
        if(FB_CAN_PROMPT)
            message(STATUS "   Or disable prompts with: -DINTERACTIVE_BACKEND_SETUP=OFF")
        else()
            message(STATUS "   Interactive prompts are suppressed for this non-interactive configure run")
        endif()
    endif()
    message(STATUS "══════════════════════════════════════════════════════")
    message(STATUS "")
endif()

# Build OpenBLAS from source if requested and not already found with threading
if(BUILD_OPENBLAS_FROM_SOURCE AND NOT OPENBLAS_BUILT_FROM_SOURCE)
    if(NOT OpenBLAS_FOUND OR INTERACTIVE_BACKEND_SETUP)
        build_openblas_from_source()
    endif()
endif()
