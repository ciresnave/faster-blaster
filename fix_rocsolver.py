#!/usr/bin/env python3
"""
Automated cuSOLVER → rocSOLVER API conversion
Converts LAPACK functions from cuSOLVER style to rocSOLVER style
"""

import re
import sys

# Read the file
with open('src/backends/gpu/rocblas_trait_impl.c', 'r') as f:
    content = f.read()

# Count before
before_count = len(re.findall(r'_bufferSize', content))
print(f"Before: {before_count} _bufferSize calls")

# PATTERN 1: getrf-style (has devInfo, needs workspace removed)
# rocsolver_Xgetrf(handle, m, n, A, lda, ipiv, info)
for dtype in [('s', 'float'), ('d', 'double'), ('c', 'rocblas_float_complex'), ('z', 'rocblas_double_complex')]:
    prefix, ctype = dtype
    
    # Fix getrf - remove bufferSize query and workspace
    pattern = rf'''(static int rocblas_{prefix}getrf_impl.*?\{{[^}}]*?rocblas_status status;\s+if \(stream\) \{{[^}}]*?\}})
\s+// Query workspace size
\s+status = rocsolver_{prefix}getrf_bufferSize\([^;]+;\s+if \(status != rocblas_status_success\) return -1;
\s+// Allocate workspace and device info
\s+hipMalloc\(\(void\*\*\)&workspace, lwork \* sizeof\({ctype}\)\);
\s+hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);
\s+// Perform LU factorization
\s+status = rocsolver_{prefix}getrf\(ctx->solver_handle, m, n, \({ctype}\*\)a, lda,
\s+workspace, \(int\*\)ipiv, devInfo\);'''
    
    replacement = rf'''\1
    
    // Allocate device info (rocSOLVER manages workspace internally)
    hipMalloc((void**)&devInfo, sizeof(int));
    
    // Perform LU factorization
    status = rocsolver_{prefix}getrf(ctx->solver_handle, m, n, ({ctype}*)a, lda,
                               (int*)ipiv, devInfo);'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# PATTERN 2: potrf-style (Cholesky)
for dtype in [('s', 'float'), ('d', 'double'), ('c', 'rocblas_float_complex'), ('z', 'rocblas_double_complex')]:
    prefix, ctype = dtype
    
    pattern = rf'''status = rocsolver_{prefix}potrf_bufferSize\(ctx->solver_handle, fill, n,
\s+\({ctype}\*\)a, lda, &lwork\);
\s+if \(status != rocblas_status_success\) return -1;
\s+hipMalloc\(\(void\*\*\)&workspace, lwork \* sizeof\({ctype}\)\);
\s+hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);
\s+status = rocsolver_{prefix}potrf\(ctx->solver_handle, fill, n, \({ctype}\*\)a, lda,
\s+workspace, lwork, devInfo\);'''
    
    replacement = rf'''hipMalloc((void**)&devInfo, sizeof(int));
    
    // rocSOLVER potrf manages workspace internally
    status = rocsolver_{prefix}potrf(ctx->solver_handle, fill, n, ({ctype}*)a, lda, devInfo);'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# PATTERN 3: potrs-style (no devInfo in rocSOLVER)
for dtype in [('s', 'float'), ('d', 'double'), ('c', 'rocblas_float_complex'), ('z', 'rocblas_double_complex')]:
    prefix, ctype = dtype
    
    # Remove devInfo from potrs
    content = re.sub(rf'rocsolver_{prefix}potrs\((.*?)\), devInfo\)', rf'rocsolver_{prefix}potrs(\1))', content)
    
    # Fix the return statement for potrs
    pattern = rf'''(static int rocblas_{prefix}potrs_impl.*?rocblas_status status;.*?stream.*?\}})
\s+hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);
\s+(status = rocsolver_{prefix}potrs.*?\);)
\s+int info;
\s+hipMemcpy\(&info, devInfo, sizeof\(int\), hipMemcpyDeviceToHost\);
\s+hipFree\(devInfo\);
\s+return \(status == rocblas_status_success && info == 0\)'''
    
    replacement = rf'''\1
    
    // rocSOLVER potrs doesn't use devInfo
    \2
    
    return (status == rocblas_status_success)'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# PATTERN 4: geqrf-style (QR)
for dtype in [('s', 'float'), ('d', 'double'), ('c', 'rocblas_float_complex'), ('z', 'rocblas_double_complex')]:
    prefix, ctype = dtype
    
    pattern = rf'''status = rocsolver_{prefix}geqrf_bufferSize\(ctx->solver_handle, m, n,
\s+\({ctype}\*\)a, lda, &lwork\);
\s+if \(status != rocblas_status_success\) return -1;
\s+hipMalloc\(\(void\*\*\)&workspace, lwork \* sizeof\({ctype}\)\);
\s+hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);
\s+status = rocsolver_{prefix}geqrf\(ctx->solver_handle, m, n, \({ctype}\*\)a, lda,
\s+\({ctype}\*\)tau, workspace, lwork, devInfo\);'''
    
    replacement = rf'''hipMalloc((void**)&devInfo, sizeof(int));
    
    // rocSOLVER geqrf manages workspace internally
    status = rocsolver_{prefix}geqrf(ctx->solver_handle, m, n, ({ctype}*)a, lda,
                              ({ctype}*)tau, devInfo);'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# PATTERN 5: gesvd-style (SVD) - more complex
for dtype in [('s', 'float'), ('d', 'double'), ('c', 'rocblas_float_complex'), ('z', 'rocblas_double_complex')]:
    prefix, ctype = dtype
    real_type = 'float' if prefix in ['s', 'c'] else 'double'
    
    pattern = rf'''status = rocsolver_{prefix}gesvd_bufferSize\(ctx->solver_handle, m, n, &lwork\);
\s+if \(status != rocblas_status_success\) return -1;
\s+(?:int minmn = \(m < n\) \? m : n;\s+)?hipMalloc\(\(void\*\*\)&workspace, lwork \* sizeof\({ctype}\)\);
\s+(?:hipMalloc\(\(void\*\*\)&rwork, 5 \* minmn \* sizeof\({real_type}\)\);\s+)?hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);
\s+status = rocsolver_{prefix}gesvd\(ctx->solver_handle, jobu, jobvt, m, n,
\s+\({ctype}\*\)a, lda, \({real_type}\*\)s,
\s+\({ctype}\*\)u, ldu, \({ctype}\*\)vt, ldvt,
\s+workspace, lwork, (?:rwork|NULL), devInfo\);'''
    
    rwork_alloc = f'int minmn = (m < n) ? m : n;\n    hipMalloc((void**)&rwork, 5 * minmn * sizeof({real_type}));\n    ' if prefix in ['c', 'z'] else ''
    rwork_param = 'rwork' if prefix in ['c', 'z'] else 'NULL'
    
    replacement = rf'''{rwork_alloc}hipMalloc((void**)&devInfo, sizeof(int));
    
    // rocSOLVER gesvd uses different API (left_svect, right_svect instead of jobu, jobvt)
    status = rocsolver_{prefix}gesvd(ctx->solver_handle, jobu, jobvt, m, n,
                              ({ctype}*)a, lda, ({real_type}*)s,
                              ({ctype}*)u, ldu, ({ctype}*)vt, ldvt,
                              {rwork_param}, devInfo);'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# PATTERN 6: syevd/heevd-style (eigenvalues)
for func in [('ssyevd', 'float'), ('dsyevd', 'double'), ('cheevd', 'rocblas_float_complex'), ('zheevd', 'rocblas_double_complex')]:
    fname, ctype = func
    real_type = 'float' if fname[0] == 's' or fname[0] == 'c' else 'double'
    
    pattern = rf'''status = rocsolver_{fname}_bufferSize\(ctx->solver_handle, jobmode, fill, n,
\s+\({ctype}\*\)a, lda, \({real_type}\*\)w, &lwork\);
\s+if \(status != rocblas_status_success\) return -1;
\s+hipMalloc\(\(void\*\*\)&workspace, lwork \* sizeof\({ctype}\)\);
\s+hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);
\s+status = rocsolver_{fname}\(ctx->solver_handle, jobmode, fill, n,
\s+\({ctype}\*\)a, lda, \({real_type}\*\)w,
\s+workspace, lwork, devInfo\);'''
    
    replacement = rf'''hipMalloc((void**)&devInfo, sizeof(int));
    
    // rocSOLVER {fname} manages workspace internally
    status = rocsolver_{fname}(ctx->solver_handle, jobmode, fill, n,
                              ({ctype}*)a, lda, ({real_type}*)w,
                              NULL, devInfo);  // E parameter for off-diagonal (can be NULL)'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Remove unused workspace variables
content = re.sub(r'\s+int lwork;\n\s+(?:float|double|rocblas_float_complex|rocblas_double_complex)\* workspace;', '', content)
content = re.sub(r'\s+hipFree\(workspace\);', '', content)

# Write back
with open('src/backends/gpu/rocblas_trait_impl.c', 'w') as f:
    f.write(content)

# Count after
after_count = len(re.findall(r'_bufferSize', content))
print(f"After: {after_count} _bufferSize calls")
print(f"Fixed: {before_count - after_count} functions")

sys.exit(0 if after_count == 0 else 1)
