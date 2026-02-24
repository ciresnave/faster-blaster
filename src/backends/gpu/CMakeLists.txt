cmake_minimum_required(VERSION 3.20)
project(onemkl_trait CXX)

# This CMakeLists.txt is designed to be built separately with Intel DPC++ compiler
# Usage: icx-cl or dpcpp compiler must be available

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Intel oneAPI components
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(OneMKL REQUIRED)

# Also find SYCL/DPC++ compiler includes
find_path(SYCL_INCLUDE_DIR sycl/sycl.hpp
    PATHS
        "$ENV{ONEAPI_ROOT}/compiler/latest"
        "C:/Program Files (x86)/Intel/oneAPI/compiler/latest"
        "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3"
        "/opt/intel/oneapi/compiler/latest"
    PATH_SUFFIXES include
)

find_path(TBB_INCLUDE_DIR tbb/tbb.h
    PATHS
        "$ENV{ONEAPI_ROOT}/tbb/latest"
        "C:/Program Files (x86)/Intel/oneAPI/tbb/latest"
        "/opt/intel/oneapi/tbb/latest"
    PATH_SUFFIXES include
)

# Create the oneMKL trait library
add_library(onemkl_trait STATIC
    onemkl_trait_impl.cpp
)

target_include_directories(onemkl_trait PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../include
    ${SYCL_INCLUDE_DIR}
    ${ONEMKL_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIR}
)

# Add SYCL compilation flag
target_compile_options(onemkl_trait PRIVATE
    /Qstd=c++20
    -fsycl
)

# Link libraries (use discovered oneMKL libraries)
target_link_libraries(onemkl_trait
    ${ONEMKL_LIBRARIES}
    tbb12
    OpenCL
)

# Installation
install(TARGETS onemkl_trait
    ARCHIVE DESTINATION lib
)
