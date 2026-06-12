#!/usr/bin/env python3
"""
Convert cuBLAS implementation to rocBLAS implementation
Reads cublas_trait_impl.c and generates complete rocblas_trait_impl.c
"""

import re

def convert_line(line):
    """Convert a single line from cuBLAS to rocBLAS"""
    # Function names
    line = line.replace('cublas_', 'rocblas_')
    line = line.replace('CUBLAS_', 'ROCBLAS_')
    
    # CUDA to HIP
    line = line.replace('cudaStream_t', 'hipStream_t')
    line = line.replace('cudaMalloc', 'hipMalloc')
    line = line.replace('cudaFree', 'hipFree')
    line = line.replace('cudaMemcpy', 'hipMemcpy')
    line = line.replace('cudaMemcpyHostToDevice', 'hipMemcpyHostToDevice')
    line = line.replace('cudaMemcpyDeviceToHost', 'hipMemcpyDeviceToHost')
    line = line.replace('cudaMemcpyDeviceToDevice', 'hipMemcpyDeviceToDevice')
    line = line.replace('cudaStreamCreate', 'hipStreamCreate')
    line = line.replace('cudaStreamDestroy', 'hipStreamDestroy')
    line = line.replace('cudaStreamSynchronize', 'hipStreamSynchronize')
    line = line.replace('cudaGetDevice', 'hipGetDevice')
    line = line.replace('cudaSetDevice', 'hipSetDevice')
    line = line.replace('cudaGetDeviceProperties', 'hipGetDeviceProperties')
    line = line.replace('cudaDeviceProp', 'hipDeviceProp_t')
    line = line.replace('cudaGetErrorString', 'hipGetErrorString')
    line = line.replace('cudaSuccess', 'hipSuccess')
    line = line.replace('cudaError_t', 'hipError_t')
    
    # Complex types
    line = line.replace('cuComplex', 'rocblas_float_complex')
    line = line.replace('cuDoubleComplex', 'rocblas_double_complex')
    
    # cuBLAS/cuSOLVER core API
    line = line.replace('cublasCreate', 'rocblas_create_handle')
    line = line.replace('cublasDestroy', 'rocblas_destroy_handle')
    line = line.replace('cublasSetStream', 'rocblas_set_stream')
    line = line.replace('cusolverDnCreate', 'rocblas_create_handle')
    line = line.replace('cusolverDnDestroy', 'rocblas_destroy_handle')
    line = line.replace('cusolverDnSetStream', 'rocblas_set_stream')
    line = line.replace('cudaDeviceSynchronize', 'hipDeviceSynchronize')
    
    # Context structure fields
    line = line.replace('rocblas_handle rocblas_handle', 'rocblas_handle blas_handle')
    line = line.replace('rocblas_handle cusolver_handle', 'rocblas_handle solver_handle')
    line = line.replace('->rocblas_handle', '->blas_handle')
    line = line.replace('->cusolver_handle', '->solver_handle')
    
    # Status types and constants
    line = line.replace('cusolverStatus_t', 'rocblas_status')
    line = line.replace('CUSOLVER_STATUS_SUCCESS', 'rocblas_status_success')
    line = line.replace('CUBLAS_STATUS_SUCCESS', 'rocblas_status_success')
    line = line.replace('ROCBLAS_STATUS_SUCCESS', 'rocblas_status_success')
    
    # Constants - uppercase to lowercase enumsline = line.replace('ROCBLAS_OP_N', 'rocblas_operation_none')
    line = line.replace('ROCBLAS_OP_T', 'rocblas_operation_transpose')
    line = line.replace('ROCBLAS_OP_C', 'rocblas_operation_conjugate_transpose')
    
    # cuBLAS API to rocBLAS API - handle capitalization
    # SetStream
    line = line.replace('rocblas_set_stream', 'rocblas_set_stream')
    
    # Operations (need to be case-sensitive)
    # Level 1
    for op in ['Saxpy', 'Daxpy', 'Sscal', 'Dscal', 'Scopy', 'Dcopy', 'Sswap', 'Dswap',
               'Sdot', 'Ddot', 'Snrm2', 'Dnrm2', 'Sasum', 'Dasum', 'Isamax', 'Idamax',
               'Caxpy', 'Zaxpy', 'Cscal', 'Zscal', 'Csscal', 'Zdscal', 'Ccopy', 'Zcopy',
               'Cswap', 'Zswap', 'Cdotu', 'Zdotu', 'Cdotc', 'Zdotc',
               'Scnrm2', 'Dznrm2', 'Scasum', 'Dzasum', 'Icamax', 'Izamax',
               'Srot', 'Drot', 'Crot', 'Zrot', 'Srotg', 'Drotg', 'Crotg', 'Zrotg',
               'Srotm', 'Drotm', 'Srotmg', 'Drotmg']:
        line = line.replace(f'cublas{op}', f'rocblas_{op.lower()}')
    
    # Level 2
    for op in ['Sgemv', 'Dgemv', 'Cgemv', 'Zgemv',
               'Sgbmv', 'Dgbmv', 'Cgbmv', 'Zgbmv',
               'Ssymv', 'Dsymv', 'Chemv', 'Zhemv',
               'Ssbmv', 'Dsbmv', 'Chbmv', 'Zhbmv',
               'Sspmv', 'Dspmv', 'Chpmv', 'Zhpmv',
               'Strmv', 'Dtrmv', 'Ctrmv', 'Ztrmv',
               'Stbmv', 'Dtbmv', 'Ctbmv', 'Ztbmv',
               'Stpmv', 'Dtpmv', 'Ctpmv', 'Ztpmv',
               'Strsv', 'Dtrsv', 'Ctrsv', 'Ztrsv',
               'Stbsv', 'Dtbsv', 'Ctbsv', 'Ztbsv',
               'Stpsv', 'Dtpsv', 'Ctpsv', 'Ztpsv',
               'Sger', 'Dger', 'Cgeru', 'Zgeru', 'Cgerc', 'Zgerc',
               'Ssyr', 'Dsyr', 'Cher', 'Zher',
               'Sspr', 'Dspr', 'Chpr', 'Zhpr',
               'Ssyr2', 'Dsyr2', 'Cher2', 'Zher2',
               'Sspr2', 'Dspr2', 'Chpr2', 'Zhpr2',
               'Csymv', 'Zsymv', 'Csyr', 'Zsyr', 'Csyr2', 'Zsyr2']:
        line = line.replace(f'cublas{op}', f'rocblas_{op.lower()}')
    
    # Level 3
    for op in ['Sgemm', 'Dgemm', 'Cgemm', 'Zgemm',
               'Ssymm', 'Dsymm', 'Csymm', 'Zsymm', 'Chemm', 'Zhemm',
               'Ssyrk', 'Dsyrk', 'Csyrk', 'Zsyrk', 'Cherk', 'Zherk',
               'Ssyr2k', 'Dsyr2k', 'Csyr2k', 'Zsyr2k', 'Cher2k', 'Zher2k',
               'Strmm', 'Dtrmm', 'Ctrmm', 'Ztrmm',
               'Strsm', 'Dtrsm', 'Ctrsm', 'Ztrsm']:
        line = line.replace(f'cublas{op}', f'rocblas_{op.lower()}')
    
    # Constants - rocBLAS uses lowercase
    line = line.replace('CUBLAS_FILL_MODE_UPPER', 'rocblas_fill_upper')
    line = line.replace('CUBLAS_FILL_MODE_LOWER', 'rocblas_fill_lower')
    line = line.replace('CUBLAS_OP_N', 'rocblas_operation_none')
    line = line.replace('CUBLAS_OP_T', 'rocblas_operation_transpose')
    line = line.replace('CUBLAS_OP_C', 'rocblas_operation_conjugate_transpose')
    line = line.replace('CUBLAS_DIAG_NON_UNIT', 'rocblas_diagonal_non_unit')
    line = line.replace('CUBLAS_DIAG_UNIT', 'rocblas_diagonal_unit')
    line = line.replace('CUBLAS_SIDE_LEFT', 'rocblas_side_left')
    
    # cuSOLVER LAPACK functions to rocSOLVER
    # LU factorization
    for prefix in ['S', 'D', 'C', 'Z']:
        line = line.replace(f'cusolverDn{prefix}getrf_bufferSize', f'rocsolver_{prefix.lower()}getrf_bufferSize')
        line = line.replace(f'cusolverDn{prefix}getrf', f'rocsolver_{prefix.lower()}getrf')
        line = line.replace(f'cusolverDn{prefix}getrs', f'rocsolver_{prefix.lower()}getrs')
        line = line.replace(f'cusolverDn{prefix}potrf_bufferSize', f'rocsolver_{prefix.lower()}potrf_bufferSize')
        line = line.replace(f'cusolverDn{prefix}potrf', f'rocsolver_{prefix.lower()}potrf')
        line = line.replace(f'cusolverDn{prefix}potrs', f'rocsolver_{prefix.lower()}potrs')
        line = line.replace(f'cusolverDn{prefix}geqrf_bufferSize', f'rocsolver_{prefix.lower()}geqrf_bufferSize')
        line = line.replace(f'cusolverDn{prefix}geqrf', f'rocsolver_{prefix.lower()}geqrf')
        line = line.replace(f'cusolverDn{prefix}ormqr_bufferSize', f'rocsolver_{prefix.lower()}ormqr_bufferSize')
        line = line.replace(f'cusolverDn{prefix}ormqr', f'rocsolver_{prefix.lower()}ormqr')
        line = line.replace(f'cusolverDn{prefix}unmqr_bufferSize', f'rocsolver_{prefix.lower()}unmqr_bufferSize')
        line = line.replace(f'cusolverDn{prefix}unmqr', f'rocsolver_{prefix.lower()}unmqr')
        line = line.replace(f'cusolverDn{prefix}gesvd_bufferSize', f'rocsolver_{prefix.lower()}gesvd_bufferSize')
        line = line.replace(f'cusolverDn{prefix}gesvd', f'rocsolver_{prefix.lower()}gesvd')
        line = line.replace(f'cusolverDn{prefix}syevd_bufferSize', f'rocsolver_{prefix.lower()}syevd_bufferSize')
        line = line.replace(f'cusolverDn{prefix}syevd', f'rocsolver_{prefix.lower()}syevd')
        line = line.replace(f'cusolverDn{prefix}heevd_bufferSize', f'rocsolver_{prefix.lower()}heevd_bufferSize')
        line = line.replace(f'cusolverDn{prefix}heevd', f'rocsolver_{prefix.lower()}heevd')
    
    # cuSOLVER eigenvalue mode enum
    line = line.replace('cusolverEigMode_t', 'rocblas_evect')
    line = line.replace('CUSOLVER_EIG_MODE_VECTOR', 'rocblas_evect_original')
    line = line.replace('CUSOLVER_EIG_MODE_NOVECTOR', 'rocblas_evect_none')
    line = line.replace('CUBLAS_SIDE_RIGHT', 'rocblas_side_right')
    
    return line

def main():
    print("Reading cublas_trait_impl.c...")
    with open(r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\src\backends\gpu\cublas_trait_impl.c', 'r') as f:
        content = f.read()
    
    print("Converting to rocBLAS...")
    lines = content.split('\n')
    converted_lines = []
    
    in_header = True
    for line in lines:
        if in_header and '* BLAS Level 1' in line:
            in_header = False
        
        if in_header:
            # Update header
            if 'cuBLAS' in line or 'CUDA' in line:
                line = line.replace('cuBLAS', 'rocBLAS')
                line = line.replace('CUDA', 'HIP')
                line = line.replace('NVIDIA', 'AMD')
            if '#include <cuda_runtime.h>' in line:
                line = '#include <hip/hip_runtime.h>'
            if '#include <cublas_v2.h>' in line:
                line = '#include <rocblas/rocblas.h>'
            if '#include <cusolverDn.h>' in line:
                line = '#include <rocsolver/rocsolver.h>'
            if 'typedef struct {' in line:
                converted_lines.append(line)
                converted_lines.append('    rocblas_handle rocblas_handle;')
                converted_lines.append('    int device_id;')
                converted_lines.append('} rocblas_context_t;')
                # Skip the cublas typedef
                while lines and 'cublas_context_t' not in lines[0]:
                    lines.pop(0)
                if lines:
                    lines.pop(0)  # Skip the closing typedef
                continue
        
        converted_line = convert_line(line)
        converted_lines.append(converted_line)
    
    output = '\n'.join(converted_lines)
    
    # Fix any remaining issues
    output = output.replace('fb_get_cublas_trait', 'fb_get_rocblas_trait')
    output = output.replace('fb_cublas_trait', 'fb_rocblas_trait')
    
    print("Writing rocblas_trait_impl.c...")
    with open(r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\src\backends\gpu\rocblas_trait_impl_generated.c', 'w') as f:
        f.write(output)
    
    print("Done! Review rocblas_trait_impl_generated.c and merge with existing rocblas_trait_impl.c")

if __name__ == '__main__':
    main()
