# FindAOCL.cmake
# Locate AMD Optimizing CPU Libraries (AOCL)
#
# This module defines:
#  AOCL_FOUND - System has AOCL
#  AOCL_INCLUDE_DIRS - AOCL include directories
#  AOCL_LIBRARIES - Libraries needed to use AOCL (BLIS + libFLAME)
#  AOCL_VERSION - Version of AOCL found
#
# Users can help this module find AOCL by setting:
#  AOCL_ROOT - CMake cache variable or environment variable pointing to AOCL installation
#  FB_ENABLE_DEEP_SEARCH - Set to ON to search common installation directories (slower)

# Tier 1: User hints (highest priority)
set(AOCL_SEARCH_PATHS
    # CMake cache variable (set via -DAOCL_ROOT=...)
    ${AOCL_ROOT}
    # CMake cache variable (set via -DAOCL_ROOT=...)
    ${AOCL_ROOT}
    # Environment variables
    $ENV{AOCL_ROOT}
    $ENV{AOCL_HOME}
    
    # Tier 2: vcpkg installations
    "$ENV{VCPKG_ROOT}/installed/x64-windows"
    "C:/vcpkg/installed/x64-windows"
    "C:/libraries/vcpkg/installed/x64-windows"
    "C:/dev/vcpkg/installed/x64-windows"
    
    # Tier 3: Known standard locations
    # Windows AOCL installations
    "C:/Program Files/AMD/AOCL-Windows"
    "C:/Program Files/AMD/AOCL-Windows/amd-libflame"
    "C:/Program Files/AMD/AOCL/AOCL-Windows-gcc-4.2.0"
    "C:/Program Files/AMD/AOCL/AOCL-Windows-gcc-4.1.0"
    "C:/Program Files/AMD/AOCL/AOCL-Windows-gcc-4.0.0"
    "C:/Program Files/AMD/AOCL"
    "C:/AMD/AOCL"
    
    # Linux default locations
    /opt/AMD/aocl
    /opt/AMD/AOCL
    /opt/aocl
    /usr/local/aocl
    
    # Versioned paths (Linux)
    /opt/AMD/aocl/aocl-linux-gcc-4.2.0
    /opt/AMD/aocl/aocl-linux-gcc-4.1.0
    /opt/AMD/aocl/aocl-linux-gcc-4.0.0
)

# Tier 4: Deep search in common roots (opt-in, slower)
if(FB_ENABLE_DEEP_SEARCH)
    message(STATUS "AOCL: Deep search enabled, scanning common installation directories...")
    
    # Search Windows Program Files for AMD/AOCL directories
    if(WIN32)
        file(GLOB AOCL_AMD_DIRS 
            "C:/Program Files/AMD/AOCL*"
            "C:/Program Files/AMD/*/AOCL*"
        )
        list(APPEND AOCL_SEARCH_PATHS ${AOCL_AMD_DIRS})
        
        # Also search subdirectories one level deep
        foreach(dir ${AOCL_AMD_DIRS})
            file(GLOB AOCL_SUB_DIRS "${dir}/*")
            list(APPEND AOCL_SEARCH_PATHS ${AOCL_SUB_DIRS})
        endforeach()
    else()
        # Linux: search /opt and /usr/local
        file(GLOB AOCL_OPT_DIRS 
            "/opt/AMD/aocl*"
            "/opt/AMD/AOCL*"
            "/opt/aocl*"
            "/usr/local/aocl*"
        )
        list(APPEND AOCL_SEARCH_PATHS ${AOCL_OPT_DIRS})
    endif()
endif()

# Find BLIS include directory (AOCL's BLAS component)
find_path(AOCL_BLIS_INCLUDE_DIR
    NAMES blis/blis.h blis.h
    PATHS ${AOCL_SEARCH_PATHS}
    PATH_SUFFIXES amd-blis/include/LP64 amd-blis/include/ILP64 include include_LP64 include_ILP64 include/blis
    DOC "AOCL BLIS include directory"
)

# Find libFLAME include directory (AOCL's LAPACK component)
find_path(AOCL_FLAME_INCLUDE_DIR
    NAMES FLAME.h
    PATHS ${AOCL_SEARCH_PATHS}
    PATH_SUFFIXES amd-libflame/include/LP64 amd-libflame/include/ILP64 include include_LP64 include_ILP64
    DOC "AOCL libFLAME include directory"
)

# Find library directory
find_path(AOCL_LIB_DIR
    NAMES AOCL-LibBlis-Win-dll.lib AOCL-LibFlame-Win-dll.lib libblis.lib blis.lib libblis.so libblis.a
    PATHS ${AOCL_SEARCH_PATHS}
    PATH_SUFFIXES amd-blis/lib/LP64 amd-libflame/lib/LP64 amd-blis/lib/ILP64 amd-libflame/lib/ILP64 lib lib_LP64 lib_ILP64 lib64
    DOC "AOCL library directory"
)

# Find BLIS library
find_library(AOCL_BLIS_LIBRARY
    NAMES blis libblis AOCL-LibBlis-Win-MT-dll
    PATHS ${AOCL_LIB_DIR}
    NO_DEFAULT_PATH
    DOC "AOCL BLIS library"
)

# Find libFLAME library
find_library(AOCL_FLAME_LIBRARY
    NAMES flame libflame AOCL-LibFlame-Win-MT-dll
    PATHS ${AOCL_LIB_DIR}
    NO_DEFAULT_PATH
    DOC "AOCL libFLAME library"
)

# Combine libraries
set(AOCL_LIBRARIES "")
if(AOCL_BLIS_LIBRARY)
    list(APPEND AOCL_LIBRARIES ${AOCL_BLIS_LIBRARY})
endif()
if(AOCL_FLAME_LIBRARY)
    list(APPEND AOCL_LIBRARIES ${AOCL_FLAME_LIBRARY})
endif()

# Combine include directories
set(AOCL_INCLUDE_DIRS "")
if(AOCL_BLIS_INCLUDE_DIR)
    list(APPEND AOCL_INCLUDE_DIRS ${AOCL_BLIS_INCLUDE_DIR})
endif()
if(AOCL_FLAME_INCLUDE_DIR AND NOT AOCL_FLAME_INCLUDE_DIR STREQUAL AOCL_BLIS_INCLUDE_DIR)
    list(APPEND AOCL_INCLUDE_DIRS ${AOCL_FLAME_INCLUDE_DIR})
endif()

# Try to extract version from directory name or version file
if(AOCL_LIB_DIR)
    # Extract from path like "/opt/AMD/aocl/aocl-linux-gcc-4.2.0"
    string(REGEX MATCH "aocl.*-([0-9]+\\.[0-9]+\\.[0-9]+)" AOCL_VERSION_MATCH "${AOCL_LIB_DIR}")
    if(AOCL_VERSION_MATCH)
        set(AOCL_VERSION ${CMAKE_MATCH_1})
    endif()
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AOCL
    REQUIRED_VARS AOCL_BLIS_LIBRARY AOCL_INCLUDE_DIRS
    VERSION_VAR AOCL_VERSION
    FAIL_MESSAGE "Could not find AOCL. Try one of:
  1. Set AOCL_ROOT: cmake -DAOCL_ROOT=/path/to/aocl ..
  2. Set environment variable: export AOCL_ROOT=/path/to/aocl
  3. Enable deep search: cmake -DFB_ENABLE_DEEP_SEARCH=ON ..
  4. Install via package manager (vcpkg, etc.)"
)

# Set output variables
if(AOCL_FOUND)
    # Print found information
    message(STATUS "Found AOCL:")
    if(AOCL_BLIS_INCLUDE_DIR)
        message(STATUS "  BLIS Include: ${AOCL_BLIS_INCLUDE_DIR}")
    endif()
    if(AOCL_FLAME_INCLUDE_DIR)
        message(STATUS "  libFLAME Include: ${AOCL_FLAME_INCLUDE_DIR}")
    endif()
    message(STATUS "  BLIS Library: ${AOCL_BLIS_LIBRARY}")
    if(AOCL_FLAME_LIBRARY)
        message(STATUS "  libFLAME Library: ${AOCL_FLAME_LIBRARY}")
    else()
        message(STATUS "  libFLAME: Not found (LAPACK operations unavailable)")
    endif()
    if(AOCL_VERSION)
        message(STATUS "  Version: ${AOCL_VERSION}")
    endif()
endif()

mark_as_advanced(
    AOCL_BLIS_INCLUDE_DIR
    AOCL_FLAME_INCLUDE_DIR
    AOCL_LIB_DIR
    AOCL_BLIS_LIBRARY
    AOCL_FLAME_LIBRARY
    AOCL_INCLUDE_DIRS
    AOCL_LIBRARIES
)
