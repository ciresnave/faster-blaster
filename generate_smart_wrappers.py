#!/usr/bin/env python3
"""
Generate smart wrappers for all Phase 1 BLAS operations

This script generates the remaining 36 smart wrappers using the
device memory manager, following the pattern established by sgemv/dgemv.
"""

# Phase 1 BLAS operations (Level 1, 2, 3)
OPERATIONS = [
    # Level 1: Vector operations
    {
        'name': 'sdot',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
            ('const float*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(float)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'float',
        'return_code': '''
    float result = 0.0f;
    /* TODO: Call trait sdot and return result */
    return result;
'''
    },
    {
        'name': 'ddot',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
            ('const double*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(double)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'double',
        'return_code': '''
    double result = 0.0;
    /* TODO: Call trait ddot and return result */
    return result;
'''
    },
    {
        'name': 'snrm2',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'float',
        'return_code': '''
    float result = 0.0f;
    /* TODO: Call trait snrm2 and return result */
    return result;
'''
    },
    {
        'name': 'dnrm2',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'double',
        'return_code': '''
    double result = 0.0;
    /* TODO: Call trait dnrm2 and return result */
    return result;
'''
    },
    {
        'name': 'sasum',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'float',
        'return_code': '''
    float result = 0.0f;
    /* TODO: Call trait sasum and return result */
    return result;
'''
    },
    {
        'name': 'dasum',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'double',
        'return_code': '''
    double result = 0.0;
    /* TODO: Call trait dasum and return result */
    return result;
'''
    },
    {
        'name': 'isamax',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'int',
        'return_code': '''
    int result = 0;
    /* TODO: Call trait isamax and return result */
    return result;
'''
    },
    {
        'name': 'idamax',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'int',
        'return_code': '''
    int result = 0;
    /* TODO: Call trait idamax and return result */
    return result;
'''
    },
    {
        'name': 'sswap',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
            ('float*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(float)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'void',
        'outputs': ['x', 'y'],
    },
    {
        'name': 'dswap',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
            ('double*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(double)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'void',
        'outputs': ['x', 'y'],
    },
    {
        'name': 'scopy',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
            ('float*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(float)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'void',
        'outputs': ['y'],
    },
    {
        'name': 'dcopy',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
            ('double*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(double)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'void',
        'outputs': ['y'],
    },
    {
        'name': 'saxpy',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('float', 'alpha', 'alpha', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
            ('float*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(float)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'void',
        'outputs': ['y'],
    },
    {
        'name': 'daxpy',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('double', 'alpha', 'alpha', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
            ('double*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(double)'),
            ('int', 'incy', 'incy', False),
        ],
        'return_type': 'void',
        'outputs': ['y'],
    },
    {
        'name': 'sscal',
        'type': 'float',
        'params': [
            ('int', 'n', 'n', False),
            ('float', 'alpha', 'alpha', False),
            ('float*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'void',
        'outputs': ['x'],
    },
    {
        'name': 'dscal',
        'type': 'double',
        'params': [
            ('int', 'n', 'n', False),
            ('double', 'alpha', 'alpha', False),
            ('double*', 'x', 'x', 'calculate_vector_size(n, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
        ],
        'return_type': 'void',
        'outputs': ['x'],
    },
    
    # Level 2: Matrix-vector operations (already have sgemv/dgemv)
    {
        'name': 'sger',
        'type': 'float',
        'params': [
            ('int', 'm', 'm', False),
            ('int', 'n', 'n', False),
            ('float', 'alpha', 'alpha', False),
            ('const float*', 'x', 'x', 'calculate_vector_size(m, incx) * sizeof(float)'),
            ('int', 'incx', 'incx', False),
            ('const float*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(float)'),
            ('int', 'incy', 'incy', False),
            ('float*', 'a', 'a', 'calculate_matrix_size_col_major(m, n, lda) * sizeof(float)'),
            ('int', 'lda', 'lda', False),
        ],
        'return_type': 'void',
        'outputs': ['a'],
    },
    {
        'name': 'dger',
        'type': 'double',
        'params': [
            ('int', 'm', 'm', False),
            ('int', 'n', 'n', False),
            ('double', 'alpha', 'alpha', False),
            ('const double*', 'x', 'x', 'calculate_vector_size(m, incx) * sizeof(double)'),
            ('int', 'incx', 'incx', False),
            ('const double*', 'y', 'y', 'calculate_vector_size(n, incy) * sizeof(double)'),
            ('int', 'incy', 'incy', False),
            ('double*', 'a', 'a', 'calculate_matrix_size_col_major(m, n, lda) * sizeof(double)'),
            ('int', 'lda', 'lda', False),
        ],
        'return_type': 'void',
        'outputs': ['a'],
    },
    
    # Level 3: Matrix-matrix operations
    {
        'name': 'sgemm',
        'type': 'float',
        'params': [
            ('char', 'transa', 'transa', False),
            ('char', 'transb', 'transb', False),
            ('int', 'm', 'm', False),
            ('int', 'n', 'n', False),
            ('int', 'k', 'k', False),
            ('float', 'alpha', 'alpha', False),
            ('const float*', 'a', 'a', 'calculate_matrix_size_col_major(transa == \'N\' ? m : k, transa == \'N\' ? k : m, lda) * sizeof(float)'),
            ('int', 'lda', 'lda', False),
            ('const float*', 'b', 'b', 'calculate_matrix_size_col_major(transb == \'N\' ? k : n, transb == \'N\' ? n : k, ldb) * sizeof(float)'),
            ('int', 'ldb', 'ldb', False),
            ('float', 'beta', 'beta', False),
            ('float*', 'c', 'c', 'calculate_matrix_size_col_major(m, n, ldc) * sizeof(float)'),
            ('int', 'ldc', 'ldc', False),
        ],
        'return_type': 'void',
        'outputs': ['c'],
    },
    {
        'name': 'dgemm',
        'type': 'double',
        'params': [
            ('char', 'transa', 'transa', False),
            ('char', 'transb', 'transb', False),
            ('int', 'm', 'm', False),
            ('int', 'n', 'n', False),
            ('int', 'k', 'k', False),
            ('double', 'alpha', 'alpha', False),
            ('const double*', 'a', 'a', 'calculate_matrix_size_col_major(transa == \'N\' ? m : k, transa == \'N\' ? k : m, lda) * sizeof(double)'),
            ('int', 'lda', 'lda', False),
            ('const double*', 'b', 'b', 'calculate_matrix_size_col_major(transb == \'N\' ? k : n, transb == \'N\' ? n : k, ldb) * sizeof(double)'),
            ('int', 'ldb', 'ldb', False),
            ('double', 'beta', 'beta', False),
            ('double*', 'c', 'c', 'calculate_matrix_size_col_major(m, n, ldc) * sizeof(double)'),
            ('int', 'ldc', 'ldc', False),
        ],
        'return_type': 'void',
        'outputs': ['c'],
    },
]

def generate_wrapper(op):
    """Generate smart wrapper code for a single operation"""
    name = op['name']
    return_type = op.get('return_type', 'void')
    params = op['params']
    outputs = op.get('outputs', [])
    
    # Generate parameter declarations
    param_decls = ', '.join([f"{p[0]} {p[1]}" for p in params])
    
    # Generate function signature
    code = f"\n/**\n * @brief Smart wrapper for {name}\n */\n"
    code += f"{return_type} cublas_{name}_smart_wrapper({param_decls}) {{\n"
    
    # Get device pointers for inputs
    for p in params:
        param_type, param_name, trait_param, size_expr = p
        if size_expr and size_expr != False:
            # This is a pointer parameter
            is_const = 'const' in param_type
            if is_const:
                code += f"    /* Get device buffer for input {param_name} */\n"
                code += f"    fb_gpu_ptr_t d_{param_name};\n"
                code += f"    size_t size_{param_name} = {size_expr};\n"
                code += f"    if (get_device_buffer_input({param_name}, size_{param_name}, &d_{param_name}) != 0) {{\n"
                if return_type != 'void':
                    code += f"        return ({return_type})0;\n"
                else:
                    code += f"        return;\n"
                code += f"    }}\n\n"
    
    # Get device pointers for outputs
    for out_name in outputs:
        # Find the parameter info
        for p in params:
            if p[1] == out_name:
                size_expr = p[3]
                code += f"    /* Get device buffer for output {out_name} */\n"
                code += f"    fb_gpu_ptr_t d_{out_name};\n"
                code += f"    size_t size_{out_name} = {size_expr};\n"
                code += f"    if (get_device_buffer_output({out_name}, size_{out_name}, &d_{out_name}) != 0) {{\n"
                if return_type != 'void':
                    code += f"        return ({return_type})0;\n"
                else:
                    code += f"        return;\n"
                code += f"    }}\n\n"
                break
    
    # Call the GPU trait function
    code += f"    /* Call GPU trait {name} */\n"
    code += f"    /* TODO: Implement trait call for {name} */\n\n"
    
    # Mark outputs as dirty
    for out_name in outputs:
        code += f"    /* Mark {out_name} as dirty */\n"
        code += f"    mark_output_dirty({out_name}, d_{out_name});\n\n"
    
    # Release references
    for p in params:
        param_type, param_name, trait_param, size_expr = p
        if size_expr and size_expr != False:
            code += f"    /* Release {param_name} */\n"
            code += f"    release_device_buffer(d_{param_name});\n\n"
    
    # Return statement for non-void functions
    if return_type != 'void':
        if 'return_code' in op:
            code += op['return_code']
        else:
            code += f"    return ({return_type})0;\n"
    
    code += "}\n"
    return code

def main():
    """Generate all smart wrappers"""
    print("/* ============================================================================")
    print(" * Generated Smart Wrappers for Phase 1 BLAS Operations")
    print(" * ========================================================================== */")
    
    for op in OPERATIONS:
        print(generate_wrapper(op))

if __name__ == '__main__':
    main()
