cmake_minimum_required(VERSION 3.15)

# Simple GEMM example
add_executable(simple_gemm simple_gemm.c)
target_link_libraries(simple_gemm PRIVATE faster-blaster)

# Example: Vector operations
add_executable(vector_ops vector_ops.c)
target_link_libraries(vector_ops PRIVATE faster-blaster)
