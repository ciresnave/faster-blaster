#!/usr/bin/env python3
"""
Convert cuBLAS implementation to oneMKL implementation
Reads cublas_trait_impl.c and generates complete onemkl_trait_impl.cpp
"""

import re
import sys

# Read cuBLAS file
with open('src/backends/gpu/cublas_trait_impl.c', 'r') as f:
    cublas_content = f.read()

# Extract all BLAS function implementations
pattern = r'static void cublas_(\w+)_impl\(([^)]+(?:\([^)]+\))?[^)]*)\)\s*\{([^}]+(?:\{[^}]+\}[^}]*)*)\}'
matches = re.findall(pattern, cublas_content, re.MULTILINE | re.DOTALL)

print(f"Found {len(matches)} cuBLAS implementations")

# Conversion rules
type_conversions = {
    'cuComplex': 'std::complex<float>',
    'cuDoubleComplex': 'std::complex<double>',
    'cublas_context_t': 'onemkl_context_t',
    'cublas_handle': 'sycl_queue',
    'cublasFillMode_t': 'oneapi::mkl::uplo',
    'cublasOperation_t': 'oneapi::mkl::transpose',
    'cublasDiagType_t': 'oneapi::mkl::diag',
    'cublasSideMode_t': 'oneapi::mkl::side',
}

api_conversions = {
    'cublasSaxpy': 'oneapi::mkl::blas::axpy',
    'cublasDaxpy': 'oneapi::mkl::blas::axpy',
    'cublasCaxpy': 'oneapi::mkl::blas::axpy',
    'cublasZaxpy': 'oneapi::mkl::blas::axpy',
    # Add more as needed
}

def convert_function(name, params, body):
    """Convert a single cuBLAS function to oneMKL"""
    # Convert function name
    onemkl_name = name
    
    # Convert parameters
    onemkl_params = params
    for old, new in type_conversions.items():
        onemkl_params = onemkl_params.replace(old, new)
    
    # Convert body
    onemkl_body = body
    
    # Replace context access
    onemkl_body = onemkl_body.replace('cublas_context_t* ctx = (cublas_context_t*)handle;',
                                       'onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);')
    
    # Replace stream handling
    onemkl_body = onemkl_body.replace('cublasSetStream(ctx->cublas_handle, (cudaStream_t)stream);',
                                       '')
    
    # Add SYCL queue selection
    queue_selection = "    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;"
    
    # Convert cuBLAS API calls to oneMKL
    for cublas_api, onemkl_api in api_conversions.items():
        onemkl_body = onemkl_body.replace(cublas_api, onemkl_api)
    
    # Wrap in try-catch
    onemkl_impl = f"""static void onemkl_{onemkl_name}_impl({onemkl_params}) {{
{queue_selection}
    try {{
        // TODO: Convert cuBLAS call to oneMKL
{onemkl_body}
    }} catch (const oneapi::mkl::exception& e) {{
        fprintf(stderr, "oneMKL {onemkl_name} error: %s\\n", e.what());
    }}
}}
"""
    
    return onemkl_impl

# Print first few as examples
for i, (name, params, body) in enumerate(matches[:5]):
    print(f"\n=== {i+1}. {name} ===")
    print(f"Params: {params[:80]}...")
    print(f"Body snippet: {body[:100].strip()}...")
    
print(f"\nTotal implementations to convert: {len(matches)}")
print("\nNote: Full automatic conversion is complex. Manual conversion recommended.")
print("Suggestion: Use this script to extract operation names, then use templates.")
