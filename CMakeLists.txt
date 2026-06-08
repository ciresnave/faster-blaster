cmake_minimum_required(VERSION 3.15)

# Enforce Ninja generator — the VS project generator produces MSBuild artifacts
# that are incompatible with our clang-cl/LLD toolchain configuration.
# Use:  cmake --preset clang   (or clang-release)
if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(FATAL_ERROR
        "This project requires the Ninja generator.\n"
        "Run:  cmake --preset clang          (Debug)\n"
        "  or  cmake --preset clang-release  (Release)\n"
        "Current generator: ${CMAKE_GENERATOR}")
endif()

if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING
        "Windows builds must use the x64 vcpkg triplet" FORCE)
endif()

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

# Bootstrap the x64 Windows SDK/MSVC environment before project().
# CMake's compiler sanity checks for clang-cl + lld-link run inside project(),
# so missing LIB/INCLUDE search paths must be fixed here rather than later.
if(WIN32)
    set(FB_WIN_BOOTSTRAP_LIB_DIRS "")
    set(FB_WIN_BOOTSTRAP_INCLUDE_DIRS "")
    set(FB_WIN_BOOTSTRAP_MISSING "")
    set(FB_PROGRAM_FILES_X86 "C:/Program Files (x86)")
    set(FB_VCTOOLS_DIR "")
    set(FB_UCRT_ROOT "")
    set(FB_UCRT_VERSION "")
    set(FB_WINDOWS_SDK_ROOT "")
    set(FB_WINDOWS_SDK_VERSION "")

    if(DEFINED ENV{ProgramFiles} AND NOT "$ENV{ProgramFiles}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{ProgramFiles}" FB_PROGRAM_FILES_DIR)
        get_filename_component(FB_PROGRAM_FILES_ROOT "${FB_PROGRAM_FILES_DIR}" DIRECTORY)
        if(FB_PROGRAM_FILES_ROOT)
            set(FB_PROGRAM_FILES_X86 "${FB_PROGRAM_FILES_ROOT}/Program Files (x86)")
        endif()
    endif()

    if(DEFINED ENV{VCToolsInstallDir} AND NOT "$ENV{VCToolsInstallDir}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" FB_VCTOOLS_DIR)
    else()
        find_program(FB_VSWHERE
            NAMES vswhere
            PATHS
                "${FB_PROGRAM_FILES_X86}/Microsoft Visual Studio/Installer"
                "$ENV{ProgramFiles}/Microsoft Visual Studio/Installer"
            NO_DEFAULT_PATH)
        if(NOT FB_VSWHERE)
            find_program(FB_VSWHERE NAMES vswhere)
        endif()

        if(FB_VSWHERE)
            execute_process(
                COMMAND "${FB_VSWHERE}" -latest -products *
                        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
                        -property installationPath
                OUTPUT_VARIABLE FB_VS_INSTALL_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(FB_VS_INSTALL_DIR)
                file(TO_CMAKE_PATH "${FB_VS_INSTALL_DIR}" FB_VS_INSTALL_DIR)
                file(GLOB FB_VCTOOLS_CANDIDATES LIST_DIRECTORIES TRUE
                    "${FB_VS_INSTALL_DIR}/VC/Tools/MSVC/*")
                if(FB_VCTOOLS_CANDIDATES)
                    list(SORT FB_VCTOOLS_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
                    list(GET FB_VCTOOLS_CANDIDATES 0 FB_VCTOOLS_DIR)
                endif()
            endif()
        endif()
    endif()

    if(FB_VCTOOLS_DIR)
        if(EXISTS "${FB_VCTOOLS_DIR}/lib/x64")
            list(APPEND FB_WIN_BOOTSTRAP_LIB_DIRS "${FB_VCTOOLS_DIR}/lib/x64")
        endif()
        if(EXISTS "${FB_VCTOOLS_DIR}/ATLMFC/lib/x64")
            list(APPEND FB_WIN_BOOTSTRAP_LIB_DIRS "${FB_VCTOOLS_DIR}/ATLMFC/lib/x64")
        endif()
        if(EXISTS "${FB_VCTOOLS_DIR}/include")
            list(APPEND FB_WIN_BOOTSTRAP_INCLUDE_DIRS "${FB_VCTOOLS_DIR}/include")
        endif()
        if(EXISTS "${FB_VCTOOLS_DIR}/ATLMFC/include")
            list(APPEND FB_WIN_BOOTSTRAP_INCLUDE_DIRS "${FB_VCTOOLS_DIR}/ATLMFC/include")
        endif()
    endif()

    if(DEFINED ENV{UniversalCRTSdkDir} AND DEFINED ENV{UCRTVersion}
       AND NOT "$ENV{UniversalCRTSdkDir}" STREQUAL "" AND NOT "$ENV{UCRTVersion}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{UniversalCRTSdkDir}" FB_UCRT_ROOT)
        set(FB_UCRT_VERSION "$ENV{UCRTVersion}")
    endif()

    if(DEFINED ENV{WindowsSdkDir} AND DEFINED ENV{WindowsSDKLibVersion}
       AND NOT "$ENV{WindowsSdkDir}" STREQUAL "" AND NOT "$ENV{WindowsSDKLibVersion}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{WindowsSdkDir}" FB_WINDOWS_SDK_ROOT)
        set(FB_WINDOWS_SDK_VERSION "$ENV{WindowsSDKLibVersion}")
    endif()

    if(NOT FB_WINDOWS_SDK_ROOT)
        if(EXISTS "${FB_PROGRAM_FILES_X86}/Windows Kits/10/Lib")
            set(FB_WINDOWS_SDK_ROOT "${FB_PROGRAM_FILES_X86}/Windows Kits/10")
        elseif(DEFINED ENV{ProgramFiles} AND EXISTS "$ENV{ProgramFiles}/Windows Kits/10/Lib")
            file(TO_CMAKE_PATH "$ENV{ProgramFiles}/Windows Kits/10" FB_WINDOWS_SDK_ROOT)
        endif()
    endif()

    if(NOT FB_UCRT_ROOT)
        set(FB_UCRT_ROOT "${FB_WINDOWS_SDK_ROOT}")
    endif()

    if(FB_WINDOWS_SDK_ROOT AND NOT FB_WINDOWS_SDK_VERSION)
        set(FB_WINDOWS_SDK_VERSION_CANDIDATES "")
        file(GLOB FB_WINDOWS_SDK_LIB_DIRS LIST_DIRECTORIES TRUE
            "${FB_WINDOWS_SDK_ROOT}/Lib/10.*")
        foreach(FB_WINDOWS_SDK_LIB_DIR ${FB_WINDOWS_SDK_LIB_DIRS})
            if(EXISTS "${FB_WINDOWS_SDK_LIB_DIR}/ucrt/x64" AND EXISTS "${FB_WINDOWS_SDK_LIB_DIR}/um/x64")
                list(APPEND FB_WINDOWS_SDK_VERSION_CANDIDATES "${FB_WINDOWS_SDK_LIB_DIR}")
            endif()
        endforeach()
        if(FB_WINDOWS_SDK_VERSION_CANDIDATES)
            list(SORT FB_WINDOWS_SDK_VERSION_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
            list(GET FB_WINDOWS_SDK_VERSION_CANDIDATES 0 FB_WINDOWS_SDK_LIB_DIR)
            get_filename_component(FB_WINDOWS_SDK_VERSION "${FB_WINDOWS_SDK_LIB_DIR}" NAME)
        endif()
    endif()

    if(NOT FB_UCRT_VERSION)
        set(FB_UCRT_VERSION "${FB_WINDOWS_SDK_VERSION}")
    endif()

    if(FB_UCRT_ROOT AND FB_UCRT_VERSION
       AND EXISTS "${FB_UCRT_ROOT}/Include/${FB_UCRT_VERSION}/ucrt")
        list(APPEND FB_WIN_BOOTSTRAP_INCLUDE_DIRS "${FB_UCRT_ROOT}/Include/${FB_UCRT_VERSION}/ucrt")
    endif()

    foreach(FB_WINDOWS_SDK_INCLUDE_SUBDIR shared um winrt cppwinrt)
        if(FB_WINDOWS_SDK_ROOT AND FB_WINDOWS_SDK_VERSION
           AND EXISTS "${FB_WINDOWS_SDK_ROOT}/Include/${FB_WINDOWS_SDK_VERSION}/${FB_WINDOWS_SDK_INCLUDE_SUBDIR}")
            list(APPEND FB_WIN_BOOTSTRAP_INCLUDE_DIRS
                "${FB_WINDOWS_SDK_ROOT}/Include/${FB_WINDOWS_SDK_VERSION}/${FB_WINDOWS_SDK_INCLUDE_SUBDIR}")
        endif()
    endforeach()

    if(FB_UCRT_ROOT AND FB_UCRT_VERSION
       AND EXISTS "${FB_UCRT_ROOT}/Lib/${FB_UCRT_VERSION}/ucrt/x64")
        list(APPEND FB_WIN_BOOTSTRAP_LIB_DIRS "${FB_UCRT_ROOT}/Lib/${FB_UCRT_VERSION}/ucrt/x64")
    endif()

    if(FB_WINDOWS_SDK_ROOT AND FB_WINDOWS_SDK_VERSION
       AND EXISTS "${FB_WINDOWS_SDK_ROOT}/Lib/${FB_WINDOWS_SDK_VERSION}/um/x64")
        list(APPEND FB_WIN_BOOTSTRAP_LIB_DIRS "${FB_WINDOWS_SDK_ROOT}/Lib/${FB_WINDOWS_SDK_VERSION}/um/x64")
    endif()

    if(NOT FB_VCTOOLS_DIR)
        list(APPEND FB_WIN_BOOTSTRAP_MISSING "MSVC build tools were not found")
    endif()
    if(NOT FB_WINDOWS_SDK_ROOT)
        list(APPEND FB_WIN_BOOTSTRAP_MISSING "Windows SDK root was not found")
    endif()
    if(FB_WINDOWS_SDK_ROOT AND NOT FB_WINDOWS_SDK_VERSION)
        list(APPEND FB_WIN_BOOTSTRAP_MISSING "No Windows SDK version with x64 um/ucrt libraries was found")
    endif()

    if(FB_WIN_BOOTSTRAP_MISSING)
        string(JOIN "\n  - " FB_WIN_BOOTSTRAP_MISSING_TEXT ${FB_WIN_BOOTSTRAP_MISSING})
        message(FATAL_ERROR
            "Windows x64 toolchain bootstrap failed before project().\n"
            "Configure is using clang-cl/lld-link, so CMake's compiler test cannot run without MSVC + Windows SDK x64 libraries.\n"
            "Resolved values:\n"
            "  VCToolsInstallDir='${FB_VCTOOLS_DIR}'\n"
            "  UniversalCRTSdkDir='${FB_UCRT_ROOT}'\n"
            "  UCRTVersion='${FB_UCRT_VERSION}'\n"
            "  WindowsSdkDir='${FB_WINDOWS_SDK_ROOT}'\n"
            "  WindowsSDKLibVersion='${FB_WINDOWS_SDK_VERSION}'\n"
            "Missing components:\n  - ${FB_WIN_BOOTSTRAP_MISSING_TEXT}\n"
            "Fix one of the following:\n"
            "  1. Install Visual Studio Build Tools with Desktop development with C++.\n"
            "  2. Install the Windows 10/11 SDK.\n"
            "  3. Configure from an x64 developer shell, or let this bootstrap resolve the tools automatically.")
    endif()

    list(REMOVE_DUPLICATES FB_WIN_BOOTSTRAP_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES FB_WIN_BOOTSTRAP_LIB_DIRS)

    if(FB_WIN_BOOTSTRAP_INCLUDE_DIRS)
        list(JOIN FB_WIN_BOOTSTRAP_INCLUDE_DIRS ";" FB_WIN_BOOTSTRAP_INCLUDE_ENV)
        if(DEFINED ENV{INCLUDE} AND NOT "$ENV{INCLUDE}" STREQUAL "")
            set(ENV{INCLUDE} "${FB_WIN_BOOTSTRAP_INCLUDE_ENV};$ENV{INCLUDE}")
        else()
            set(ENV{INCLUDE} "${FB_WIN_BOOTSTRAP_INCLUDE_ENV}")
        endif()
        message(STATUS "[WINDOWS TOOLCHAIN] Added x64 INCLUDE paths for clang-cl")
    endif()

    if(FB_WIN_BOOTSTRAP_LIB_DIRS)
        list(JOIN FB_WIN_BOOTSTRAP_LIB_DIRS ";" FB_WIN_BOOTSTRAP_LIB_ENV)
        if(DEFINED ENV{LIB} AND NOT "$ENV{LIB}" STREQUAL "")
            set(ENV{LIB} "${FB_WIN_BOOTSTRAP_LIB_ENV};$ENV{LIB}")
        else()
            set(ENV{LIB} "${FB_WIN_BOOTSTRAP_LIB_ENV}")
        endif()

        foreach(FB_WIN_LINK_INIT_VAR
                CMAKE_EXE_LINKER_FLAGS_INIT
                CMAKE_SHARED_LINKER_FLAGS_INIT
                CMAKE_MODULE_LINKER_FLAGS_INIT
                CMAKE_STATIC_LINKER_FLAGS_INIT)
            set(_fb_win_link_init_value "${${FB_WIN_LINK_INIT_VAR}}")
            string(REGEX REPLACE "(^| )/machine:[^ ]+" " "
                _fb_win_link_init_value "${_fb_win_link_init_value}")
            string(STRIP "${_fb_win_link_init_value}" _fb_win_link_init_value)
            set(${FB_WIN_LINK_INIT_VAR} "${_fb_win_link_init_value}")
        endforeach()

        set(FB_WIN_BOOTSTRAP_LINK_FLAGS "")
        foreach(FB_WIN_LIB_DIR ${FB_WIN_BOOTSTRAP_LIB_DIRS})
            string(APPEND FB_WIN_BOOTSTRAP_LINK_FLAGS " /libpath:\"${FB_WIN_LIB_DIR}\"")
        endforeach()
        string(APPEND FB_WIN_BOOTSTRAP_LINK_FLAGS " /machine:x64")
        set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT}${FB_WIN_BOOTSTRAP_LINK_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT}${FB_WIN_BOOTSTRAP_LINK_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS_INIT "${CMAKE_MODULE_LINKER_FLAGS_INIT}${FB_WIN_BOOTSTRAP_LINK_FLAGS}")
        set(CMAKE_STATIC_LINKER_FLAGS_INIT "${CMAKE_STATIC_LINKER_FLAGS_INIT} /machine:x64")
        message(STATUS "[WINDOWS TOOLCHAIN] Added x64 LIB paths for lld-link")
    endif()
endif()

function(fb_append_unique_flag flag_var flag_value)
    set(_fb_flag_value "${${flag_var}}")
    if(NOT " ${_fb_flag_value} " MATCHES "(^| )${flag_value}( |$)")
        string(APPEND _fb_flag_value " ${flag_value}")
        string(STRIP "${_fb_flag_value}" _fb_flag_value)
        set(${flag_var} "${_fb_flag_value}" PARENT_SCOPE)
    endif()
endfunction()

function(fb_append_unique_cached_flag flag_var flag_value)
    set(_fb_flag_value "${${flag_var}}")
    if(NOT " ${_fb_flag_value} " MATCHES "(^| )${flag_value}( |$)")
        string(APPEND _fb_flag_value " ${flag_value}")
        string(STRIP "${_fb_flag_value}" _fb_flag_value)
        set(${flag_var} "${_fb_flag_value}" CACHE STRING
            "Windows Clang linker flags normalized for LLVM LLD" FORCE)
    endif()
endfunction()

set(FB_FORCE_LLD_WITH_CLANG OFF)
if(WIN32)
    foreach(FB_WIN_COMPILER_CANDIDATE
            "${CMAKE_C_COMPILER}"
            "${CMAKE_CXX_COMPILER}"
            "$ENV{CC}"
            "$ENV{CXX}")
        if(FB_WIN_COMPILER_CANDIDATE MATCHES "(^|[\\/])clang(\\+\\+|-cl)?(\\.exe)?$")
            set(FB_FORCE_LLD_WITH_CLANG ON)
            break()
        endif()
    endforeach()
endif()

if(FB_FORCE_LLD_WITH_CLANG)
    unset(CMAKE_LINKER CACHE)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.29")
        set(CMAKE_LINKER_TYPE LLD CACHE STRING
            "Prefer LLVM lld-link when Clang drives the Windows link step" FORCE)
    endif()
    foreach(FB_WIN_LLD_INIT_VAR
            CMAKE_EXE_LINKER_FLAGS_INIT
            CMAKE_SHARED_LINKER_FLAGS_INIT
            CMAKE_MODULE_LINKER_FLAGS_INIT)
        fb_append_unique_flag(${FB_WIN_LLD_INIT_VAR} "-fuse-ld=lld")
    endforeach()
endif()

project(faster-blaster VERSION 0.1.0 LANGUAGES C CXX)

if(WIN32)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        unset(CMAKE_LINKER CACHE)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.29")
            set(CMAKE_LINKER_TYPE LLD CACHE STRING
                "Prefer LLVM lld-link when Clang drives the Windows link step" FORCE)
        endif()
        foreach(FB_WIN_LLD_FLAG_VAR
                CMAKE_EXE_LINKER_FLAGS
                CMAKE_SHARED_LINKER_FLAGS
                CMAKE_MODULE_LINKER_FLAGS)
            fb_append_unique_cached_flag(${FB_WIN_LLD_FLAG_VAR} "-fuse-ld=lld")
        endforeach()
    endif()

    function(fb_force_windows_x64_machine_flag flag_var)
        set(_fb_flag_value "${${flag_var}}")
        string(REGEX REPLACE "(^| )/machine:[^ ]+" " " _fb_flag_value "${_fb_flag_value}")
        string(STRIP "${_fb_flag_value}" _fb_flag_value)
        if(_fb_flag_value)
            set(_fb_flag_value "${_fb_flag_value} /machine:x64")
        else()
            set(_fb_flag_value "/machine:x64")
        endif()
        set(${flag_var} "${_fb_flag_value}" CACHE STRING
            "Windows x64 linker flags normalized for clang-cl/Ninja" FORCE)
    endfunction()

    foreach(FB_WIN_LINK_FLAG_VAR
            CMAKE_EXE_LINKER_FLAGS
            CMAKE_SHARED_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS
            CMAKE_STATIC_LINKER_FLAGS)
        fb_force_windows_x64_machine_flag(${FB_WIN_LINK_FLAG_VAR})
    endforeach()
endif()

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
option(ENABLE_OXIBLAS "Enable OxiBLAS backend (pure-Rust, requires liboxiblas_ffi)" ON)
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
set(FB_ENABLE_OPENCL OFF)
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_INCLUDE_DIRS}")
    set(FB_ENABLE_OPENCL ON)
    add_definitions(-DFB_ENABLE_OPENCL)
else()
    message(STATUS "OpenCL not found. OpenCL/CLBlast support will be disabled.")
endif()

# Find CLBlast (OpenCL BLAS library)
# vcpkg uses CLBlast (capital C and B) for the package name
find_package(CLBlast CONFIG QUIET)
if(NOT CLBlast_FOUND)
    # Try lowercase as fallback
    find_package(clblast CONFIG QUIET)
endif()

if(OpenCL_FOUND AND (CLBlast_FOUND OR clblast_FOUND))
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
                message(STATUS "  CUDA language remains disabled with clang-cl on Windows; cuBLAS links against CUDAToolkit::cublas directly")
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

# OxiBLAS Detection (for OxiBLAS plugin — pure-Rust BLAS/LAPACK)
set(OXIBLAS_CPU_AVAILABLE OFF)
if(ENABLE_OXIBLAS)
    find_package(OxiBLAS QUIET)
    if(OXIBLAS_FOUND)
        set(OXIBLAS_CPU_AVAILABLE ON)
        message(STATUS "✓ OxiBLAS ${OXIBLAS_VERSION} detected (EXPERIMENTAL)")
        message(STATUS "  OxiBLAS plugin will be enabled")
    else()
        message(STATUS "✗ OxiBLAS not found - OxiBLAS plugin disabled")
        message(STATUS "  Build oxiblas-ffi: cargo build --release -p oxiblas-ffi")
        set(ENABLE_OXIBLAS OFF)
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
message(STATUS "  OxiBLAS:        ${OXIBLAS_CPU_AVAILABLE} (EXPERIMENTAL)")
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
    # Op-ID → vtable field mapping + ext_ops[] sync (2266-op dispatch)
    src/core/op_vtable_map.c
    src/core/vtable_wrappers_gemm.c
    src/core/vtable_wrappers_normalization.c
    src/core/vtable_wrappers_reduction.c
    src/core/vtable_dispatchers_gemm.c
    src/core/vtable_wrappers_batched.c
    src/core/vtable_wrappers_strided_to_array.c
    src/core/vtable_wrappers_precision_promotion.c
    src/core/ranked_dispatch.c
    # Auto-detect: dlsym-based ext_ops[] fill from symbol tables
    src/core/backend_auto_detect.c
    # Cross-convention thunks: Fortran↔CBLAS per-op static thunk arrays
    src/core/conv_thunks.c
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
    # ScaLAPACK / PBLAS dynamic-load dispatch layer
    src/scalapack/scalapack_dispatch.c
)

# Benchmark system sources (Phase 3.0)
set(BENCHMARK_SOURCES
    src/benchmark_classify.c
    src/benchmark_fingerprint.c
    src/benchmark_cache.c
    src/benchmark_execute.c
    src/dispatch_tables.c
    src/op_chain.c
    src/core/exec_dag.c
    src/auto_benchmark.c
    src/core/data_tracker.c
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

# Symbol tables for fb_auto_populate_ext_ops() — dlsym name maps per backend family.
# These are always compiled; each backend's get_vtable() calls into the tables it needs.
set(SYM_TABLE_SOURCES
    src/backends/sym_tables/lapacke_sym_table.c
    src/backends/sym_tables/cblas_ext_sym_table.c
    src/backends/sym_tables/mkl_ext_sym_table.c
    src/backends/sym_tables/scalapack_sym_table.c
    src/backends/sym_tables/cublas_sym_table.c
    src/backends/sym_tables/rocblas_sym_table.c
)

# Backend sources
set(BACKEND_SOURCES
    ${SYM_TABLE_SOURCES}
    src/backends/reference.c
    src/backends/backend_loader.c
    src/backends/backend_matcher.c
    src/backends/cpu/reference_trait_impl.c
)

# Plugin sources (new architecture)
set(PLUGIN_SOURCES
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
    list(APPEND BACKEND_SOURCES
        src/backends/openblas_backend.c
        src/backends/cpu/openblas_trait_impl.c
    )
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_openblas.c)
    message(STATUS "Adding OpenBLAS CPU backend source")
endif()

if(AOCL_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/aocl_backend.c)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_aocl_blis.c)
    message(STATUS "Adding AMD AOCL CPU backend source")
endif()

if(ACCELERATE_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/accelerate_backend.c)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_accelerate.c)
    message(STATUS "Adding Apple Accelerate backend source")
endif()

if(BLIS_CPU_AVAILABLE)
    list(APPEND BACKEND_SOURCES src/backends/cpu/blis_trait_impl.c)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_standard_blis.c)
endif()

if(OXIBLAS_CPU_AVAILABLE)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_oxiblas.c)
endif()

if(MKL_CPU_AVAILABLE)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_mkl.c)
endif()

# cuBLAS: pure dynamic loading — no CUDA headers required at compile time
file(GLOB CUBLAS_SOURCES src/backends/gpu/*cublas*.c)
# Exclude trait_impl/vtable_wrappers which statically link CUDA symbols
list(FILTER CUBLAS_SOURCES EXCLUDE REGEX "trait_impl\.c$|vtable_wrappers\.c$|smart.*\.c$")
if(CUDA_AVAILABLE)
    list(APPEND BACKEND_SOURCES ${CUBLAS_SOURCES})
endif()

# rocBLAS: pure dynamic loading — no HIP/ROCm headers required at compile time
# Only rocblas_backend.c uses dynamic loading; legacy step1/trait_impl files need HIP headers
file(GLOB ROCBLAS_SOURCES src/backends/gpu/*rocblas*.c)
list(FILTER ROCBLAS_SOURCES EXCLUDE REGEX "trait_impl.*\\.c$|step[0-9]+\\.c$")
if(ROCM_AVAILABLE)
    list(APPEND BACKEND_SOURCES ${ROCBLAS_SOURCES})
    message(STATUS "rocBLAS backend included (dynamic loading, deferred to runtime availability)")
endif()

if(CUDA_AVAILABLE)
    list(APPEND PLUGIN_SOURCES
        src/backends/gpu/cublas_trait_impl.c
        src/plugins/plugin_cublas.c
        src/plugins/cublas_vtable_wrappers.c
        src/plugins/cublas_smart_wrappers.c
        src/plugins/cublas_complete_smart_wrappers.c
        src/plugins/cublas_blas_level1_smart.c
        src/plugins/cublas_blas_level2_smart.c
        src/plugins/cublas_blas_level3_smart.c
    )
endif()

if(ROCM_AVAILABLE)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_rocblas.c)
endif()

if(ONEMKL_AVAILABLE)
    # Add oneMKL GPU backend trait implementations
    file(GLOB ONEMKL_SOURCES src/backends/gpu/*onemkl*.cpp)
    list(APPEND BACKEND_SOURCES ${ONEMKL_SOURCES})
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_onemkl.c)
endif()

if(METAL_AVAILABLE)
    list(APPEND PLUGIN_SOURCES src/plugins/plugin_metal.c)
endif()

if(CLBLAST_AVAILABLE)
    # Add CLBlast GPU backend trait implementation
    file(GLOB CLBLAST_SOURCES src/backends/gpu/*clblast*.c)
    list(APPEND BACKEND_SOURCES ${CLBLAST_SOURCES})
    list(APPEND PLUGIN_SOURCES
        src/plugins/plugin_clblast.c
        src/plugins/clblast_vtable_wrappers.c
        src/plugins/plugin_clblas.c
    )
    message(STATUS "Adding CLBlast backend trait implementation")
endif()

if(ENABLE_AOCL)
    list(APPEND BACKEND_SOURCES src/backends/aocl_backend.c)
endif()

# Remove backend/plugin sources that do not match the final resolved configuration.
if(NOT MKL_CPU_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES src/backends/mkl_backend.c)
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_mkl.c)
endif()

if(NOT OPENBLAS_CPU_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES
        src/backends/openblas_backend.c
        src/backends/cpu/openblas_trait_impl.c
    )
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_openblas.c)
endif()

if(NOT OXIBLAS_CPU_AVAILABLE)
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_oxiblas.c)
endif()

if(NOT BLIS_CPU_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES
        src/backends/blis_backend.c
        src/backends/cpu/blis_trait_impl.c
    )
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_standard_blis.c)
endif()

if(NOT AOCL_CPU_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES src/backends/aocl_backend.c)
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_aocl_blis.c)
endif()

if(NOT ACCELERATE_CPU_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES src/backends/accelerate_backend.c)
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_accelerate.c)
endif()

if(NOT CUDA_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES ${CUBLAS_SOURCES})
    list(REMOVE_ITEM PLUGIN_SOURCES
        src/backends/gpu/cublas_trait_impl.c
        src/plugins/plugin_cublas.c
        src/plugins/cublas_vtable_wrappers.c
        src/plugins/cublas_smart_wrappers.c
        src/plugins/cublas_complete_smart_wrappers.c
        src/plugins/cublas_blas_level1_smart.c
        src/plugins/cublas_blas_level2_smart.c
        src/plugins/cublas_blas_level3_smart.c
    )
endif()

if(NOT ROCM_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES ${ROCBLAS_SOURCES})
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_rocblas.c)
endif()

if(NOT ONEMKL_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES ${ONEMKL_SOURCES})
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_onemkl.c)
endif()

if(NOT METAL_AVAILABLE)
    list(REMOVE_ITEM PLUGIN_SOURCES src/plugins/plugin_metal.c)
endif()

if(NOT CLBLAST_AVAILABLE)
    list(REMOVE_ITEM BACKEND_SOURCES ${CLBLAST_SOURCES})
    list(REMOVE_ITEM PLUGIN_SOURCES
        src/plugins/plugin_clblast.c
        src/plugins/clblast_vtable_wrappers.c
        src/plugins/plugin_clblas.c
    )
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

# ---- faster-blaster-reference: canonical BLAS/LAPACK correctness oracle ----
# faster-blaster-reference is loaded at runtime via LoadLibrary/dlopen by
# plugin_reference.c; NO import library or link-time dependency is needed.
# fb_enumerate_and_populate() scans the DLL export table on load.
# _FBR_DLL is located here only so we can copy it to the tests directory.
set(_FBR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../faster-blaster-reference")

# ── Optional: locate the DLL for the test-copy step below ─────────────────
# MinGW/clang produces lib-prefixed DLL names (libfoo.dll), MSVC does not.
file(GLOB _FBR_DLL_CANDIDATES
    "${_FBR_DIR}/build-extended/libfaster_blaster_reference.dll"
    "${_FBR_DIR}/build-extended/faster_blaster_reference.dll")
if(_FBR_DLL_CANDIDATES)
    list(GET _FBR_DLL_CANDIDATES 0 _FBR_DLL)
    message(STATUS "Reference DLL (for test copy): ${_FBR_DLL}")
else()
    message(STATUS
        "faster-blaster-reference DLL not found — reference backend will be "
        "unavailable at runtime. Build it with:\n"
        "  cd ../faster-blaster-reference && "
        "cmake -B build-extended -G Ninja && cmake --build build-extended")
endif()

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

if(OXIBLAS_CPU_AVAILABLE)
    # OxiBLAS has no C headers — no include dir needed.
    # The plugin uses fb_enumerate_and_populate() for dynamic loading;
    # we also link the static library so that the backend adapter test works.
    target_link_libraries(faster-blaster PRIVATE ${OXIBLAS_LIBRARIES})
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_OXIBLAS)
    if(UNIX)
        # Rust FFI static libs may require the Rust standard library
        target_link_libraries(faster-blaster PRIVATE dl pthread)
    endif()
    message(STATUS "Linking OxiBLAS FFI library")
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
    target_compile_definitions(faster-blaster PRIVATE FB_ENABLE_CLBLAST)
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
    # Also copy the reference DLL so test_judge_runner can load it at runtime.
    # Preserve the original filename (lib-prefixed on MinGW/clang, no prefix on MSVC).
    if(_FBR_DLL_CANDIDATES)
        cmake_path(GET _FBR_DLL FILENAME _FBR_DLL_FILENAME)
        add_custom_command(TARGET faster-blaster POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_FBR_DLL}"
                ${CMAKE_CURRENT_BINARY_DIR}/tests/${_FBR_DLL_FILENAME}
            COMMENT "Copying ${_FBR_DLL_FILENAME} to tests directory"
        )
    endif()
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
message(STATUS "  OpenBLAS: ${OPENBLAS_CPU_AVAILABLE}")
message(STATUS "  BLIS: ${BLIS_CPU_AVAILABLE}")
message(STATUS "  Intel MKL: ${MKL_CPU_AVAILABLE}")
message(STATUS "  NVIDIA cuBLAS: ${CUDA_AVAILABLE}")
message(STATUS "  AMD rocBLAS: ${ROCM_AVAILABLE}")
message(STATUS "  Apple Accelerate: ${ACCELERATE_CPU_AVAILABLE}")
message(STATUS "===========================================")
message(STATUS "")
