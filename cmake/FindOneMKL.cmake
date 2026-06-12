# FindOneMKL.cmake
# Locate Intel oneMKL library for SYCL
#
# This module defines:
#  ONEMKL_FOUND - System has oneMKL
#  ONEMKL_INCLUDE_DIRS - oneMKL include directories
#  ONEMKL_LIBRARIES - Libraries needed to use oneMKL
#  ONEMKL_VERSION - Version of oneMKL found
#
# Users can help this module find oneMKL by setting:
#  ONEMKL_ROOT or MKLROOT - CMake cache variable or environment variable
#  FB_ENABLE_DEEP_SEARCH - Set to ON to search common installation directories (slower)

# Tier 1: User hints (highest priority)
if(WIN32 AND EXISTS "C:/Program Files (x86)/Intel/oneAPI")
    list(PREPEND ONEMKL_SEARCH_PATHS
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
        "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
    )
endif()

set(ONEMKL_SEARCH_PATHS
    # CMake cache variables
    ${ONEMKL_ROOT}
    ${MKLROOT}
    # Environment variables
    $ENV{ONEMKL_ROOT}
    $ENV{MKLROOT}
    $ENV{ONEAPI_ROOT}/mkl
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
    
    # Tier 2: Known standard locations
    # Windows oneAPI installation
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.1"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.0"
    "C:/Program Files (x86)/Intel/oneAPI/mkl"
    "C:/Program Files (x86)/Intel/oneAPI/2025.3/mkl"
    
    # Linux default locations
    /opt/intel/oneapi/mkl/latest
    /opt/intel/oneapi/mkl
    /usr/local/intel/oneapi/mkl
    
    # macOS default locations
    /opt/intel/oneapi/mkl/latest
    ~/intel/oneapi/mkl/latest
)

# Tier 3: Deep search in Intel/oneAPI directories
if(FB_ENABLE_DEEP_SEARCH)
    message(STATUS "oneMKL: Deep search enabled, scanning Intel oneAPI directories...")
    if(WIN32)
        file(GLOB ONEMKL_ONEAPI_DIRS 
            "C:/Program Files (x86)/Intel/oneAPI/*/mkl"
            "C:/Program Files (x86)/Intel/oneAPI/mkl/*"
        )
        list(APPEND ONEMKL_SEARCH_PATHS ${ONEMKL_ONEAPI_DIRS})
    else()
        file(GLOB ONEMKL_INTEL_DIRS 
            "/opt/intel/oneapi/*/mkl"
            "/opt/intel/oneapi/mkl/*"
        )
        list(APPEND ONEMKL_SEARCH_PATHS ${ONEMKL_INTEL_DIRS})
    endif()
endif()

# Find include directory
find_path(ONEMKL_INCLUDE_DIR
    NAMES oneapi/mkl.hpp
    PATHS ${ONEMKL_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "oneMKL include directory"
    NO_DEFAULT_PATH
)

# Also find SYCL compiler include directory (needed for SYCL headers)
find_path(SYCL_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    PATHS
        "$ENV{ONEAPI_ROOT}/compiler/latest"
        "C:/Program Files (x86)/Intel/oneAPI/compiler/latest"
        "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3"
        "C:/Program Files (x86)/Intel/oneAPI/compiler/latest"
        "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3"
        "/opt/intel/oneapi/compiler/latest"
    PATH_SUFFIXES include
    DOC "SYCL compiler include directory"
)

# Find LAPACK header
find_path(ONEMKL_LAPACK_INCLUDE_DIR
    NAMES oneapi/mkl/lapack.hpp
    PATHS ${ONEMKL_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "oneMKL LAPACK include directory"
)

# Find library directory
find_path(ONEMKL_LIB_DIR
    NAMES mkl_sycl.lib libmkl_sycl.so libmkl_sycl.a
    PATHS ${ONEMKL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/intel64 lib/x64
    DOC "oneMKL library directory"
)

# Find specific libraries
set(ONEMKL_LIB_NAMES
    mkl_sycl
    mkl_intel_lp64
    mkl_tbb_thread
    mkl_core
)

set(ONEMKL_LIBRARIES "")
foreach(lib_name ${ONEMKL_LIB_NAMES})
    find_library(ONEMKL_${lib_name}_LIBRARY
        NAMES ${lib_name}
        PATHS ${ONEMKL_LIB_DIR}
        NO_DEFAULT_PATH
    )
    if(ONEMKL_${lib_name}_LIBRARY)
        list(APPEND ONEMKL_LIBRARIES ${ONEMKL_${lib_name}_LIBRARY})
    endif()
endforeach()

# Extract version from mkl_version.h if found
if(ONEMKL_INCLUDE_DIR)
    find_file(ONEMKL_VERSION_FILE
        NAMES oneapi/mkl/version.hpp mkl_version.h
        PATHS ${ONEMKL_INCLUDE_DIR}
        NO_DEFAULT_PATH
    )
    
    if(ONEMKL_VERSION_FILE)
        file(STRINGS ${ONEMKL_VERSION_FILE} ONEMKL_VERSION_STRINGS
            REGEX "INTEL_MKL_VERSION")
        string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" ONEMKL_VERSION 
            "${ONEMKL_VERSION_STRINGS}")
    endif()
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OneMKL
    REQUIRED_VARS ONEMKL_INCLUDE_DIR ONEMKL_LIBRARIES
    VERSION_VAR ONEMKL_VERSION    FAIL_MESSAGE "Could not find Intel oneMKL. Try one of:
  1. Set MKLROOT: cmake -DMKLROOT=/path/to/mkl ..
  2. Set environment variable: export MKLROOT=/path/to/mkl
  3. Enable deep search: cmake -DFB_ENABLE_DEEP_SEARCH=ON ..
  4. Install Intel oneAPI: https://www.intel.com/content/www/us/en/developer/tools/oneapi/")

# Set output variables
if(ONEMKL_FOUND)
    set(ONEMKL_INCLUDE_DIRS ${ONEMKL_INCLUDE_DIR})
    if(SYCL_INCLUDE_DIR)
        list(APPEND ONEMKL_INCLUDE_DIRS ${SYCL_INCLUDE_DIR})
    endif()
    
    # Print found information
    message(STATUS "Found oneMKL:")
    message(STATUS "  Include: ${ONEMKL_INCLUDE_DIR}")
    if(SYCL_INCLUDE_DIR)
        message(STATUS "  SYCL Include: ${SYCL_INCLUDE_DIR}")
    endif()
    message(STATUS "  Libraries: ${ONEMKL_LIBRARIES}")
    if(ONEMKL_VERSION)
        message(STATUS "  Version: ${ONEMKL_VERSION}")
    endif()
endif()

mark_as_advanced(
    ONEMKL_INCLUDE_DIR
    ONEMKL_LAPACK_INCLUDE_DIR
    ONEMKL_LIB_DIR
    ONEMKL_LIBRARIES
)
