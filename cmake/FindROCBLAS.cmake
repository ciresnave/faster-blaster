# FindROCBLAS.cmake
# Locate AMD rocBLAS and rocSOLVER libraries for ROCm/HIP
#
# This module defines:
#  ROCBLAS_FOUND - System has rocBLAS
#  ROCBLAS_INCLUDE_DIRS - rocBLAS include directories
#  ROCBLAS_LIBRARIES - Libraries needed to use rocBLAS
#  ROCBLAS_VERSION - Version of rocBLAS found
#  ROCM_ROOT - Root directory of ROCm installation

# Search paths for ROCm/rocBLAS
set(ROCBLAS_SEARCH_PATHS
    # Environment variables
    $ENV{ROCM_PATH}
    $ENV{HIP_PATH}
    
    # vcpkg installations
    "$ENV{VCPKG_ROOT}/installed/x64-windows"
    "C:/vcpkg/installed/x64-windows"
    "C:/libraries/vcpkg/installed/x64-windows"
    "C:/dev/vcpkg/installed/x64-windows"
    
    # Windows default locations
    "C:/Program Files/AMD/ROCm/6.4"
    "C:/Program Files/AMD/ROCm/6.3"
    "C:/Program Files/AMD/ROCm/6.2"
    "C:/Program Files/AMD/ROCm/6.1"
    "C:/Program Files/AMD/ROCm/6.0"
    "C:/Program Files/AMD/ROCm"
    "C:/AMD/ROCm"
    
    # Linux default locations
    /opt/rocm-6.4
    /opt/rocm-6.3
    /opt/rocm-6.2
    /opt/rocm-6.1
    /opt/rocm-6.0
    /opt/rocm
    /usr/local/rocm
    
    # macOS (if ROCm ever supports it)
    ~/rocm
)

# Find HIP runtime include directory
find_path(HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    PATHS ${ROCBLAS_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "HIP runtime include directory"
    NO_DEFAULT_PATH
)

# Find rocBLAS include directory
find_path(ROCBLAS_INCLUDE_DIR
    NAMES rocblas/rocblas.h rocblas.h
    PATHS ${ROCBLAS_SEARCH_PATHS}
    PATH_SUFFIXES include include/rocblas
    DOC "rocBLAS include directory"
    NO_DEFAULT_PATH
)

# Find rocSOLVER include directory (optional but recommended)
find_path(ROCSOLVER_INCLUDE_DIR
    NAMES rocsolver/rocsolver.h rocsolver.h
    PATHS ${ROCBLAS_SEARCH_PATHS}
    PATH_SUFFIXES include include/rocsolver
    DOC "rocSOLVER include directory"
)

# Find library directory
find_path(ROCBLAS_LIB_DIR
    NAMES rocblas.lib librocblas.so librocblas.a
    PATHS ${ROCBLAS_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/x64 lib64
    DOC "rocBLAS library directory"
)

# Find specific libraries
set(ROCBLAS_LIB_NAMES
    rocblas
    rocsolver
    hipblas
)

set(ROCBLAS_LIBRARIES "")
foreach(lib_name ${ROCBLAS_LIB_NAMES})
    find_library(ROCBLAS_${lib_name}_LIBRARY
        NAMES ${lib_name}
        PATHS ${ROCBLAS_LIB_DIR}
        NO_DEFAULT_PATH
    )
    if(ROCBLAS_${lib_name}_LIBRARY)
        list(APPEND ROCBLAS_LIBRARIES ${ROCBLAS_${lib_name}_LIBRARY})
    endif()
endforeach()

# Determine ROCM_ROOT from found paths
if(HIP_INCLUDE_DIR)
    get_filename_component(ROCM_ROOT "${HIP_INCLUDE_DIR}" DIRECTORY)
elseif(ROCBLAS_INCLUDE_DIR)
    get_filename_component(ROCM_ROOT "${ROCBLAS_INCLUDE_DIR}" DIRECTORY)
endif()

# Extract version from rocblas-version.h if found
if(ROCBLAS_INCLUDE_DIR)
    find_file(ROCBLAS_VERSION_FILE
        NAMES rocblas-version.h
        PATHS ${ROCBLAS_INCLUDE_DIR}
        PATH_SUFFIXES rocblas
        NO_DEFAULT_PATH
    )
    
    if(ROCBLAS_VERSION_FILE)
        file(STRINGS ${ROCBLAS_VERSION_FILE} ROCBLAS_VERSION_STRINGS
            REGEX "rocblas_version")
        string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" ROCBLAS_VERSION 
            "${ROCBLAS_VERSION_STRINGS}")
    endif()
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCBLAS
    REQUIRED_VARS ROCBLAS_INCLUDE_DIR HIP_INCLUDE_DIR ROCBLAS_LIBRARIES
    VERSION_VAR ROCBLAS_VERSION
)

# Set output variables
if(ROCBLAS_FOUND)
    set(ROCBLAS_INCLUDE_DIRS ${ROCBLAS_INCLUDE_DIR} ${HIP_INCLUDE_DIR})
    if(ROCSOLVER_INCLUDE_DIR)
        list(APPEND ROCBLAS_INCLUDE_DIRS ${ROCSOLVER_INCLUDE_DIR})
    endif()
    
    # Print found information
    message(STATUS "Found rocBLAS:")
    message(STATUS "  ROCm Root: ${ROCM_ROOT}")
    message(STATUS "  HIP Include: ${HIP_INCLUDE_DIR}")
    message(STATUS "  rocBLAS Include: ${ROCBLAS_INCLUDE_DIR}")
    if(ROCSOLVER_INCLUDE_DIR)
        message(STATUS "  rocSOLVER Include: ${ROCSOLVER_INCLUDE_DIR}")
    endif()
    message(STATUS "  Libraries: ${ROCBLAS_LIBRARIES}")
    if(ROCBLAS_VERSION)
        message(STATUS "  Version: ${ROCBLAS_VERSION}")
    endif()
endif()

mark_as_advanced(
    HIP_INCLUDE_DIR
    ROCBLAS_INCLUDE_DIR
    ROCSOLVER_INCLUDE_DIR
    ROCBLAS_LIB_DIR
    ROCBLAS_LIBRARIES
    ROCM_ROOT
)
