# FindBackendLibraries.cmake
# Unified discovery system for all BLAS/LAPACK backends
#
# Usage:
#   find_backend_library(
#     NAME <backend_name>
#     HEADERS <header1> [<header2> ...]
#     LIBRARIES <lib1> [<lib2> ...]
#     SEARCH_PATHS <path1> [<path2> ...]
#     [REQUIRED]
#   )

function(find_backend_library)
    set(options REQUIRED)
    set(oneValueArgs NAME)
    set(multiValueArgs HEADERS LIBRARIES SEARCH_PATHS ENV_VARS)
    cmake_parse_arguments(BACKEND "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Add environment variable paths
    foreach(env_var ${BACKEND_ENV_VARS})
        if(DEFINED ENV{${env_var}})
            list(APPEND BACKEND_SEARCH_PATHS $ENV{${env_var}})
        endif()
    endforeach()
    
    # Find include directory
    find_path(${BACKEND_NAME}_INCLUDE_DIR
        NAMES ${BACKEND_HEADERS}
        PATHS ${BACKEND_SEARCH_PATHS}
        PATH_SUFFIXES include include/${BACKEND_NAME}
        DOC "${BACKEND_NAME} include directory"
    )
    
    # Find libraries
    set(${BACKEND_NAME}_LIBRARIES "")
    foreach(lib_name ${BACKEND_LIBRARIES})
        find_library(${BACKEND_NAME}_${lib_name}_LIBRARY
            NAMES ${lib_name}
            PATHS ${BACKEND_SEARCH_PATHS}
            PATH_SUFFIXES lib lib64 lib/intel64 lib/x64 bin
            DOC "${BACKEND_NAME} ${lib_name} library"
        )
        if(${BACKEND_NAME}_${lib_name}_LIBRARY)
            list(APPEND ${BACKEND_NAME}_LIBRARIES ${${BACKEND_NAME}_${lib_name}_LIBRARY})
        endif()
    endforeach()
    
    # Determine if found
    if(${BACKEND_NAME}_INCLUDE_DIR AND ${BACKEND_NAME}_LIBRARIES)
        set(${BACKEND_NAME}_FOUND TRUE PARENT_SCOPE)
        set(${BACKEND_NAME}_INCLUDE_DIRS ${${BACKEND_NAME}_INCLUDE_DIR} PARENT_SCOPE)
        set(${BACKEND_NAME}_LIBRARIES ${${BACKEND_NAME}_LIBRARIES} PARENT_SCOPE)
        
        message(STATUS "Found ${BACKEND_NAME}:")
        message(STATUS "  Include: ${${BACKEND_NAME}_INCLUDE_DIR}")
        message(STATUS "  Libraries: ${${BACKEND_NAME}_LIBRARIES}")
    else()
        set(${BACKEND_NAME}_FOUND FALSE PARENT_SCOPE)
        if(BACKEND_REQUIRED)
            message(FATAL_ERROR "${BACKEND_NAME} not found. Search paths: ${BACKEND_SEARCH_PATHS}")
        else()
            message(STATUS "${BACKEND_NAME} not found (optional)")
        endif()
    endif()
endfunction()

# Platform-specific default search paths
if(WIN32)
    set(DEFAULT_BACKEND_SEARCH_PATHS
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
        "C:/Program Files (x86)/Intel/oneAPI"
        "C:/AMD/ROCm"
        "C:/Program Files/OpenBLAS"
    )
elseif(APPLE)
    set(DEFAULT_BACKEND_SEARCH_PATHS
        /usr/local/cuda
        /opt/intel/oneapi
        /opt/rocm
        /usr/local/opt/openblas
        /System/Library/Frameworks/Accelerate.framework
    )
else() # Linux
    set(DEFAULT_BACKEND_SEARCH_PATHS
        /usr/local/cuda
        /opt/cuda
        /opt/intel/oneapi
        /opt/rocm
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
    )
endif()

# Export for use by other CMake files
set(BACKEND_SEARCH_PATHS ${DEFAULT_BACKEND_SEARCH_PATHS} CACHE STRING
    "Default search paths for backend libraries")
