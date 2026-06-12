# FindOpenBLAS.cmake
# Locate OpenBLAS library (portable BLAS/LAPACK)
#
# This module defines:
#  OPENBLAS_FOUND - System has OpenBLAS
#  OPENBLAS_INCLUDE_DIRS - OpenBLAS include directories
#  OPENBLAS_LIBRARIES - Libraries needed to use OpenBLAS
#  OPENBLAS_VERSION - Version of OpenBLAS found

# Search paths for OpenBLAS
set(OPENBLAS_SEARCH_PATHS
    # Build-from-source installation (highest priority)
    "${CMAKE_BINARY_DIR}/backends-install/openblas"
    "${FB_BACKENDS_INSTALL_PREFIX}/openblas"
    
    # Environment variables
    $ENV{OPENBLAS_ROOT}
    $ENV{OPENBLAS_HOME}
    
    # vcpkg installations (check environment variable first)
    "$ENV{VCPKG_ROOT}/installed/x64-windows"
    "$ENV{VCPKG_ROOT}/installed/x86-windows"
    "C:/vcpkg/installed/x64-windows"
    "C:/libraries/vcpkg/installed/x64-windows"
    "C:/dev/vcpkg/installed/x64-windows"
    
    # Windows default locations
    "C:/libraries/OpenBLAS-0.3.30-x64"
    "C:/libraries/OpenBLAS-0.3.29-x64"
    "C:/libraries/OpenBLAS-0.3.28-x64"
    "C:/Program Files/OpenBLAS"
    "C:/Program Files (x86)/OpenBLAS"
    "C:/OpenBLAS"
    "C:/msys64/mingw64"
    
    # Linux package manager locations
    /usr
    /usr/local
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /opt/OpenBLAS
    /opt/openblas
    
    # macOS Homebrew/MacPorts
    /usr/local/opt/openblas
    /opt/homebrew/opt/openblas
    /opt/local
)

# Find include directory
find_path(OPENBLAS_INCLUDE_DIR
    NAMES openblas/cblas.h cblas.h openblas_config.h
    PATHS ${OPENBLAS_SEARCH_PATHS}
    PATH_SUFFIXES include include/openblas
    DOC "OpenBLAS include directory"
)

# Find library
find_library(OPENBLAS_LIBRARY
    NAMES openblas libopenblas
    PATHS ${OPENBLAS_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 bin
    DOC "OpenBLAS library"
)

# OpenBLAS usually includes LAPACK, but check for separate LAPACK if needed
find_library(OPENBLAS_LAPACK_LIBRARY
    NAMES lapack liblapack
    PATHS ${OPENBLAS_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 bin
    DOC "LAPACK library (if separate from OpenBLAS)"
)

# Set libraries list
set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
if(OPENBLAS_LAPACK_LIBRARY AND NOT OPENBLAS_LAPACK_LIBRARY STREQUAL OPENBLAS_LIBRARY)
    list(APPEND OPENBLAS_LIBRARIES ${OPENBLAS_LAPACK_LIBRARY})
endif()

# Extract version from openblas_config.h
if(OPENBLAS_INCLUDE_DIR)
    find_file(OPENBLAS_CONFIG_FILE
        NAMES openblas_config.h
        PATHS ${OPENBLAS_INCLUDE_DIR}
        PATH_SUFFIXES openblas
        NO_DEFAULT_PATH
    )
    
    if(OPENBLAS_CONFIG_FILE)
        file(STRINGS ${OPENBLAS_CONFIG_FILE} OPENBLAS_VERSION_STRINGS
            REGEX "OPENBLAS_VERSION")
        
        # Try to extract version string
        string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" OPENBLAS_VERSION_MATCH
            "${OPENBLAS_VERSION_STRINGS}")
        if(OPENBLAS_VERSION_MATCH)
            set(OPENBLAS_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OPENBLAS_INCLUDE_DIR OPENBLAS_LIBRARY
    VERSION_VAR OPENBLAS_VERSION
)

# Set output variables
if(OPENBLAS_FOUND)
    set(OPENBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
    
    # Print found information
    message(STATUS "Found OpenBLAS:")
    message(STATUS "  Include: ${OPENBLAS_INCLUDE_DIR}")
    message(STATUS "  Library: ${OPENBLAS_LIBRARY}")
    if(OPENBLAS_LAPACK_LIBRARY AND NOT OPENBLAS_LAPACK_LIBRARY STREQUAL OPENBLAS_LIBRARY)
        message(STATUS "  LAPACK: ${OPENBLAS_LAPACK_LIBRARY}")
    else()
        message(STATUS "  LAPACK: Included in OpenBLAS")
    endif()
    if(OPENBLAS_VERSION)
        message(STATUS "  Version: ${OPENBLAS_VERSION}")
    endif()
endif()

mark_as_advanced(
    OPENBLAS_INCLUDE_DIR
    OPENBLAS_LIBRARY
    OPENBLAS_LAPACK_LIBRARY
    OPENBLAS_LIBRARIES
)
