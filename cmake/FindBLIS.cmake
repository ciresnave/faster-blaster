# FindBLIS.cmake
# Locate BLIS library (AMD-optimized BLAS)
#
# This module defines:
#  BLIS_FOUND - System has BLIS
#  BLIS_INCLUDE_DIRS - BLIS include directories
#  BLIS_LIBRARIES - Libraries needed to use BLIS
#  BLIS_VERSION - Version of BLIS found
#
# Users can help this module find BLIS by setting:
#  BLIS_ROOT - CMake cache variable or environment variable pointing to BLIS installation
#  FB_ENABLE_DEEP_SEARCH - Set to ON to search common installation directories (slower)

# Tier 1: User hints (highest priority)
set(BLIS_SEARCH_PATHS
    # CMake cache variable
    ${BLIS_ROOT}
    # CMake cache variable
    ${BLIS_ROOT}
    # Environment variables
    $ENV{BLIS_ROOT}
    $ENV{BLIS_HOME}
    
    # Tier 2: vcpkg installations
    "$ENV{VCPKG_ROOT}/installed/x64-windows"
    "C:/vcpkg/installed/x64-windows"
    "C:/libraries/vcpkg/installed/x64-windows"
    "C:/dev/vcpkg/installed/x64-windows"
    
    # Tier 3: Known standard locations
    # Windows (BLIS is part of AOCL)
    "C:/Program Files/AMD/AOCL-Windows/amd-blis"
    "C:/Program Files/AMD/AOCL-Windows"
    "C:/Program Files/BLIS"
    "C:/Program Files (x86)/BLIS"
    "C:/BLIS"
    
    # Linux package manager locations
    /usr
    /usr/local
    /opt/blis
    /opt/AMD/blis
    
    # macOS Homebrew
    /usr/local/opt/blis
    /opt/homebrew/opt/blis
)

# Tier 4: Deep search in common roots (opt-in)
if(FB_ENABLE_DEEP_SEARCH)
    message(STATUS "BLIS: Deep search enabled, scanning common installation directories...")
    if(WIN32)
        file(GLOB BLIS_AMD_DIRS "C:/Program Files/AMD/*/amd-blis" "C:/Program Files/AMD/AOCL*/amd-blis")
        list(APPEND BLIS_SEARCH_PATHS ${BLIS_AMD_DIRS})
    else()
        file(GLOB BLIS_OPT_DIRS "/opt/AMD/*/blis" "/opt/blis*")
        list(APPEND BLIS_SEARCH_PATHS ${BLIS_OPT_DIRS})
    endif()
endif()

# Find include directory
find_path(BLIS_INCLUDE_DIR
    NAMES blis/blis.h blis.h
    PATHS ${BLIS_SEARCH_PATHS}
    PATH_SUFFIXES include include/blis include/LP64 include/ILP64
    DOC "BLIS include directory"
)

# Find library
find_library(BLIS_LIBRARY
    NAMES AOCL-LibBlis-Win-dll AOCL-LibBlis-Win blis libblis
    PATHS ${BLIS_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/LP64 lib/ILP64 bin
    DOC "BLIS library"
)

# Extract version from blis.h
if(BLIS_INCLUDE_DIR)
    find_file(BLIS_HEADER_FILE
        NAMES blis.h
        PATHS ${BLIS_INCLUDE_DIR}
        PATH_SUFFIXES blis
        NO_DEFAULT_PATH
    )
    
    if(BLIS_HEADER_FILE)
        file(STRINGS ${BLIS_HEADER_FILE} BLIS_VERSION_STRINGS
            REGEX "BLIS_VERSION_STRING")
        
        # Try to extract version
        string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" BLIS_VERSION_MATCH
            "${BLIS_VERSION_STRINGS}")
        if(BLIS_VERSION_MATCH)
            set(BLIS_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLIS
    REQUIRED_VARS BLIS_INCLUDE_DIR BLIS_LIBRARY
    VERSION_VAR BLIS_VERSION
    FAIL_MESSAGE "Could not find BLIS. Try one of:
  1. Set BLIS_ROOT: cmake -DBLIS_ROOT=/path/to/blis ..
  2. Set environment variable: export BLIS_ROOT=/path/to/blis
  3. Enable deep search: cmake -DFB_ENABLE_DEEP_SEARCH=ON ..
  4. Note: BLIS is included with AMD AOCL"
)

# Set output variables
if(BLIS_FOUND)
    set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})
    set(BLIS_LIBRARIES ${BLIS_LIBRARY})
    
    # Print found information
    message(STATUS "Found BLIS:")
    message(STATUS "  Include: ${BLIS_INCLUDE_DIR}")
    message(STATUS "  Library: ${BLIS_LIBRARY}")
    if(BLIS_VERSION)
        message(STATUS "  Version: ${BLIS_VERSION}")
    endif()
    message(STATUS "  Note: BLIS provides BLAS only, LAPACK may be needed separately")
endif()

mark_as_advanced(
    BLIS_INCLUDE_DIR
    BLIS_LIBRARY
    BLIS_LIBRARIES
)
