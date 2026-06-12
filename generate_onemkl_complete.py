#!/usr/bin/env python3
"""
Generate complete oneMKL BLAS backend implementation
Based on cuBLAS reference patterns
"""

# Type mappings
TYPE_MAP = {
    's': 'float',
    'd': 'double',
    'c': 'std::complex<float>',
    'z': 'std::complex<double>',
}

def get_scalar_type(variant):
    """Get scalar parameter type for a variant"""
    if variant in ['c', 'z']:
        return 'const void*'
    return TYPE_MAP[variant]

def get_deref(variant):
    """Get dereference operator for scalar (complex needs *, scalars don't)"""
    return '*' if variant in ['c', 'z'] else ''

# Level 1 BLAS operations
level1_ops = [
    # axpy: y = alpha*x + y (already implemented - 4 variants)
    
    # scal: x = alpha*x (6 variants - already implemented)
    
    # copy: y = x (4 variants)
    {
        'name': 'copy',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy",
        'mkl_call': lambda v: f"copy(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<{TYPE_MAP[v]}*>(y), incy)"
    },
    
    # swap: swap x and y (4 variants)
    {
        'name': 'swap',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy",
        'mkl_call': lambda v: f"swap(*queue, n, static_cast<{TYPE_MAP[v]}*>(x), incx, static_cast<{TYPE_MAP[v]}*>(y), incy)"
    },
    
    # dot: result = x^T * y (6 variants)
    {
        'name': 'dot',
        'variants': ['s', 'd'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"dot(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<const {TYPE_MAP[v]}*>(y), incy, static_cast<{TYPE_MAP[v]}*>(result))"
    },
    {
        'name': 'dotu',
        'variants': ['c', 'z'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"dotu(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<const {TYPE_MAP[v]}*>(y), incy, static_cast<{TYPE_MAP[v]}*>(result))"
    },
    {
        'name': 'dotc',
        'variants': ['c', 'z'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"dotc(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<const {TYPE_MAP[v]}*>(y), incy, static_cast<{TYPE_MAP[v]}*>(result))"
    },
    
    # nrm2: Euclidean norm (4 variants)
    {
        'name': 'nrm2',
        'variants': ['s', 'd'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"nrm2(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<{TYPE_MAP[v]}*>(result))"
    },
    {
        'name': 'nrm2',
        'variants': ['c', 'z'],
        'real_result': True,
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"nrm2(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<{'float' if v=='c' else 'double'}*>(result))"
    },
    
    # asum: sum of absolute values (4 variants)
    {
        'name': 'asum',
        'variants': ['s', 'd'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"asum(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<{TYPE_MAP[v]}*>(result))"
    },
    {
        'name': 'asum',
        'variants': ['c', 'z'],
        'real_result': True,
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"asum(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<{'float' if v=='c' else 'double'}*>(result))"
    },
    
    # iamax: index of max absolute value (4 variants)
    {
        'name': 'iamax',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t result",
        'mkl_call': lambda v: f"iamax(*queue, n, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<int64_t*>(result))"
    },
]

def generate_level1_impl():
    """Generate Level 1 BLAS implementations"""
    code = []
    code.append("/* ============================================================================")
    code.append(" * BLAS Level 1 Operations - Copy/Swap/Dot/Nrm2/Asum/Iamax")
    code.append(" * ========================================================================== */")
    code.append("")
    
    for op in level1_ops:
        for v in op['variants']:
            full_name = f"{v}{op['name']}"
            params = op['params'](v)
            mkl_call = op['mkl_call'](v)
            
            code.append(f"static void onemkl_{full_name}_impl(void* handle, fb_gpu_stream_t stream, {params}) {{")
            code.append(f"    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);")
            code.append(f"    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;")
            code.append(f"    try {{")
            code.append(f"        oneapi::mkl::blas::{mkl_call};")
            code.append(f"    }} catch (const oneapi::mkl::exception& e) {{")
            code.append(f"        fprintf(stderr, \"oneMKL {full_name} error: %s\\n\", e.what());")
            code.append(f"    }}")
            code.append(f"}}")
            code.append("")
    
    return "\n".join(code)

# Level 2 BLAS operations
level2_ops = [
    # gemv: y = alpha*op(A)*x + beta*y (4 variants - sgemv already exists)
    {
        'name': 'gemv',
        'variants': ['d', 'c', 'z'],  # s already exists
        'params': lambda v: f"char trans, int m, int n, {get_scalar_type(v)} alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, {get_scalar_type(v)} beta, fb_gpu_ptr_t y, int incy",
        'mkl_call': lambda v: f"gemv(*queue, convert_transpose(trans), m, n, {get_deref(v)}static_cast<const {TYPE_MAP[v]}*>({'&alpha' if v in ['s','d'] else 'alpha'}), static_cast<const {TYPE_MAP[v]}*>(a), lda, static_cast<const {TYPE_MAP[v]}*>(x), incx, {get_deref(v)}static_cast<const {TYPE_MAP[v]}*>({'&beta' if v in ['s','d'] else 'beta'}), static_cast<{TYPE_MAP[v]}*>(y), incy)"
    },
    
    # trmv: x = op(A)*x (4 variants)
    {
        'name': 'trmv',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx",
        'mkl_call': lambda v: f"trmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const {TYPE_MAP[v]}*>(a), lda, static_cast<{TYPE_MAP[v]}*>(x), incx)"
    },
    
    # trsv: solve op(A)*x = b (4 variants)
    {
        'name': 'trsv',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx",
        'mkl_call': lambda v: f"trsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const {TYPE_MAP[v]}*>(a), lda, static_cast<{TYPE_MAP[v]}*>(x), incx)"
    },
    
    # ger/geru/gerc: A = alpha*x*y^T + A (6 variants)
    {
        'name': 'ger',
        'variants': ['s', 'd'],
        'params': lambda v: f"int m, int n, {get_scalar_type(v)} alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda",
        'mkl_call': lambda v: f"ger(*queue, m, n, alpha, static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<const {TYPE_MAP[v]}*>(y), incy, static_cast<{TYPE_MAP[v]}*>(a), lda)"
    },
    {
        'name': 'geru',
        'variants': ['c', 'z'],
        'params': lambda v: f"int m, int n, {get_scalar_type(v)} alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda",
        'mkl_call': lambda v: f"geru(*queue, m, n, *static_cast<const {TYPE_MAP[v]}*>(alpha), static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<const {TYPE_MAP[v]}*>(y), incy, static_cast<{TYPE_MAP[v]}*>(a), lda)"
    },
    {
        'name': 'gerc',
        'variants': ['c', 'z'],
        'params': lambda v: f"int m, int n, {get_scalar_type(v)} alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda",
        'mkl_call': lambda v: f"gerc(*queue, m, n, *static_cast<const {TYPE_MAP[v]}*>(alpha), static_cast<const {TYPE_MAP[v]}*>(x), incx, static_cast<const {TYPE_MAP[v]}*>(y), incy, static_cast<{TYPE_MAP[v]}*>(a), lda)"
    },
]

def generate_level2_impl():
    """Generate subset of Level 2 BLAS implementations"""
    code = []
    code.append("/* ============================================================================")
    code.append(" * BLAS Level 2 Operations - More Matrix-Vector Operations")
    code.append(" * ========================================================================== */")
    code.append("")
    
    for op in level2_ops:
        for v in op['variants']:
            full_name = f"{v}{op['name']}"
            params = op['params'](v)
            mkl_call = op['mkl_call'](v)
            
            code.append(f"static void onemkl_{full_name}_impl(void* handle, fb_gpu_stream_t stream, {params}) {{")
            code.append(f"    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);")
            code.append(f"    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;")
            code.append(f"    try {{")
            code.append(f"        oneapi::mkl::blas::{mkl_call};")
            code.append(f"    }} catch (const oneapi::mkl::exception& e) {{")
            code.append(f"        fprintf(stderr, \"oneMKL {full_name} error: %s\\n\", e.what());")
            code.append(f"    }}")
            code.append(f"}}")
            code.append("")
    
    return "\n".join(code)

# Level 3 BLAS operations
level3_ops = [
    # gemm: C = alpha*op(A)*op(B) + beta*C (4 variants - sgemm already exists)
    {
        'name': 'gemm',
        'variants': ['d', 'c', 'z'],  # s already exists
        'params': lambda v: f"char transa, char transb, int m, int n, int k, {get_scalar_type(v)} alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, {get_scalar_type(v)} beta, fb_gpu_ptr_t c, int ldc",
        'mkl_call': lambda v: f"gemm(*queue, convert_transpose(transa), convert_transpose(transb), m, n, k, {get_deref(v)}static_cast<const {TYPE_MAP[v]}*>({'&alpha' if v in ['s','d'] else 'alpha'}), static_cast<const {TYPE_MAP[v]}*>(a), lda, static_cast<const {TYPE_MAP[v]}*>(b), ldb, {get_deref(v)}static_cast<const {TYPE_MAP[v]}*>({'&beta' if v in ['s','d'] else 'beta'}), static_cast<{TYPE_MAP[v]}*>(c), ldc)"
    },
    
    # trmm: B = alpha*op(A)*B or alpha*B*op(A) (4 variants)
    {
        'name': 'trmm',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"char side, char uplo, char trans, char diag, int m, int n, {get_scalar_type(v)} alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb",
        'mkl_call': lambda v: f"trmm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, {get_deref(v)}static_cast<const {TYPE_MAP[v]}*>({'&alpha' if v in ['s','d'] else 'alpha'}), static_cast<const {TYPE_MAP[v]}*>(a), lda, static_cast<{TYPE_MAP[v]}*>(b), ldb)"
    },
    
    # trsm: solve op(A)*X = alpha*B or X*op(A) = alpha*B (4 variants)
    {
        'name': 'trsm',
        'variants': ['s', 'd', 'c', 'z'],
        'params': lambda v: f"char side, char uplo, char trans, char diag, int m, int n, {get_scalar_type(v)} alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb",
        'mkl_call': lambda v: f"trsm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, {get_deref(v)}static_cast<const {TYPE_MAP[v]}*>({'&alpha' if v in ['s','d'] else 'alpha'}), static_cast<const {TYPE_MAP[v]}*>(a), lda, static_cast<{TYPE_MAP[v]}*>(b), ldb)"
    },
]

def generate_level3_impl():
    """Generate subset of Level 3 BLAS implementations"""
    code = []
    code.append("/* ============================================================================")
    code.append(" * BLAS Level 3 Operations - More Matrix-Matrix Operations")
    code.append(" * ========================================================================== */")
    code.append("")
    
    for op in level3_ops:
        for v in op['variants']:
            full_name = f"{v}{op['name']}"
            params = op['params'](v)
            mkl_call = op['mkl_call'](v)
            
            code.append(f"static void onemkl_{full_name}_impl(void* handle, fb_gpu_stream_t stream, {params}) {{")
            code.append(f"    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);")
            code.append(f"    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;")
            code.append(f"    try {{")
            code.append(f"        oneapi::mkl::blas::{mkl_call};")
            code.append(f"    }} catch (const oneapi::mkl::exception& e) {{")
            code.append(f"        fprintf(stderr, \"oneMKL {full_name} error: %s\\n\", e.what());")
            code.append(f"    }}")
            code.append(f"}}")
            code.append("")
    
    return "\n".join(code)

if __name__ == "__main__":
    print("=== LEVEL 1 BLAS ===")
    print(generate_level1_impl())
    print("\n=== LEVEL 2 BLAS ===")
    print(generate_level2_impl())
    print("\n=== LEVEL 3 BLAS ===")
    print(generate_level3_impl())
    print("\nTotal operations generated:", 
          sum(len(op['variants']) for op in level1_ops) + 
          sum(len(op['variants']) for op in level2_ops) +
          sum(len(op['variants']) for op in level3_ops))
