/* ============================================================================
 * Generated Smart Wrappers for Phase 1 BLAS Operations
 * ========================================================================== */

/**
 * @brief Smart wrapper for sdot
 */
float cublas_sdot_smart_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (float)0;
    }

    /* Get device buffer for input y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        return (float)0;
    }

    /* Call GPU trait sdot */
    /* TODO: Implement trait call for sdot */

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);


    float result = 0.0f;
    /* TODO: Call trait sdot and return result */
    return result;
}


/**
 * @brief Smart wrapper for ddot
 */
double cublas_ddot_smart_wrapper(int n, const double* x, int incx, const double* y, int incy) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (double)0;
    }

    /* Get device buffer for input y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        return (double)0;
    }

    /* Call GPU trait ddot */
    /* TODO: Implement trait call for ddot */

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);


    double result = 0.0;
    /* TODO: Call trait ddot and return result */
    return result;
}


/**
 * @brief Smart wrapper for snrm2
 */
float cublas_snrm2_smart_wrapper(int n, const float* x, int incx) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (float)0;
    }

    /* Call GPU trait snrm2 */
    /* TODO: Implement trait call for snrm2 */

    /* Release x */
    release_device_buffer(d_x);


    float result = 0.0f;
    /* TODO: Call trait snrm2 and return result */
    return result;
}


/**
 * @brief Smart wrapper for dnrm2
 */
double cublas_dnrm2_smart_wrapper(int n, const double* x, int incx) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (double)0;
    }

    /* Call GPU trait dnrm2 */
    /* TODO: Implement trait call for dnrm2 */

    /* Release x */
    release_device_buffer(d_x);


    double result = 0.0;
    /* TODO: Call trait dnrm2 and return result */
    return result;
}


/**
 * @brief Smart wrapper for sasum
 */
float cublas_sasum_smart_wrapper(int n, const float* x, int incx) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (float)0;
    }

    /* Call GPU trait sasum */
    /* TODO: Implement trait call for sasum */

    /* Release x */
    release_device_buffer(d_x);


    float result = 0.0f;
    /* TODO: Call trait sasum and return result */
    return result;
}


/**
 * @brief Smart wrapper for dasum
 */
double cublas_dasum_smart_wrapper(int n, const double* x, int incx) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (double)0;
    }

    /* Call GPU trait dasum */
    /* TODO: Implement trait call for dasum */

    /* Release x */
    release_device_buffer(d_x);


    double result = 0.0;
    /* TODO: Call trait dasum and return result */
    return result;
}


/**
 * @brief Smart wrapper for isamax
 */
int cublas_isamax_smart_wrapper(int n, const float* x, int incx) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (int)0;
    }

    /* Call GPU trait isamax */
    /* TODO: Implement trait call for isamax */

    /* Release x */
    release_device_buffer(d_x);


    int result = 0;
    /* TODO: Call trait isamax and return result */
    return result;
}


/**
 * @brief Smart wrapper for idamax
 */
int cublas_idamax_smart_wrapper(int n, const double* x, int incx) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return (int)0;
    }

    /* Call GPU trait idamax */
    /* TODO: Implement trait call for idamax */

    /* Release x */
    release_device_buffer(d_x);


    int result = 0;
    /* TODO: Call trait idamax and return result */
    return result;
}


/**
 * @brief Smart wrapper for sswap
 */
void cublas_sswap_smart_wrapper(int n, float* x, int incx, float* y, int incy) {
    /* Get device buffer for output x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_output(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for output y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    if (get_device_buffer_output(y, size_y, &d_y) != 0) {
        return;
    }

    /* Call GPU trait sswap */
    /* TODO: Implement trait call for sswap */

    /* Mark x as dirty */
    mark_output_dirty(x, d_x);

    /* Mark y as dirty */
    mark_output_dirty(y, d_y);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

}


/**
 * @brief Smart wrapper for dswap
 */
void cublas_dswap_smart_wrapper(int n, double* x, int incx, double* y, int incy) {
    /* Get device buffer for output x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_output(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for output y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    if (get_device_buffer_output(y, size_y, &d_y) != 0) {
        return;
    }

    /* Call GPU trait dswap */
    /* TODO: Implement trait call for dswap */

    /* Mark x as dirty */
    mark_output_dirty(x, d_x);

    /* Mark y as dirty */
    mark_output_dirty(y, d_y);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

}


/**
 * @brief Smart wrapper for scopy
 */
void cublas_scopy_smart_wrapper(int n, const float* x, int incx, float* y, int incy) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for output y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    if (get_device_buffer_output(y, size_y, &d_y) != 0) {
        return;
    }

    /* Call GPU trait scopy */
    /* TODO: Implement trait call for scopy */

    /* Mark y as dirty */
    mark_output_dirty(y, d_y);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

}


/**
 * @brief Smart wrapper for dcopy
 */
void cublas_dcopy_smart_wrapper(int n, const double* x, int incx, double* y, int incy) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for output y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    if (get_device_buffer_output(y, size_y, &d_y) != 0) {
        return;
    }

    /* Call GPU trait dcopy */
    /* TODO: Implement trait call for dcopy */

    /* Mark y as dirty */
    mark_output_dirty(y, d_y);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

}


/**
 * @brief Smart wrapper for saxpy
 */
void cublas_saxpy_smart_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for output y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    if (get_device_buffer_output(y, size_y, &d_y) != 0) {
        return;
    }

    /* Call GPU trait saxpy */
    /* TODO: Implement trait call for saxpy */

    /* Mark y as dirty */
    mark_output_dirty(y, d_y);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

}


/**
 * @brief Smart wrapper for daxpy
 */
void cublas_daxpy_smart_wrapper(int n, double alpha, const double* x, int incx, double* y, int incy) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for output y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    if (get_device_buffer_output(y, size_y, &d_y) != 0) {
        return;
    }

    /* Call GPU trait daxpy */
    /* TODO: Implement trait call for daxpy */

    /* Mark y as dirty */
    mark_output_dirty(y, d_y);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

}


/**
 * @brief Smart wrapper for sscal
 */
void cublas_sscal_smart_wrapper(int n, float alpha, float* x, int incx) {
    /* Get device buffer for output x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    if (get_device_buffer_output(x, size_x, &d_x) != 0) {
        return;
    }

    /* Call GPU trait sscal */
    /* TODO: Implement trait call for sscal */

    /* Mark x as dirty */
    mark_output_dirty(x, d_x);

    /* Release x */
    release_device_buffer(d_x);

}


/**
 * @brief Smart wrapper for dscal
 */
void cublas_dscal_smart_wrapper(int n, double alpha, double* x, int incx) {
    /* Get device buffer for output x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    if (get_device_buffer_output(x, size_x, &d_x) != 0) {
        return;
    }

    /* Call GPU trait dscal */
    /* TODO: Implement trait call for dscal */

    /* Mark x as dirty */
    mark_output_dirty(x, d_x);

    /* Release x */
    release_device_buffer(d_x);

}


/**
 * @brief Smart wrapper for sger
 */
void cublas_sger_smart_wrapper(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(m, incx) * sizeof(float);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for input y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        return;
    }

    /* Get device buffer for output a */
    fb_gpu_ptr_t d_a;
    size_t size_a = calculate_matrix_size_col_major(m, n, lda) * sizeof(float);
    if (get_device_buffer_output(a, size_a, &d_a) != 0) {
        return;
    }

    /* Call GPU trait sger */
    /* TODO: Implement trait call for sger */

    /* Mark a as dirty */
    mark_output_dirty(a, d_a);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

    /* Release a */
    release_device_buffer(d_a);

}


/**
 * @brief Smart wrapper for dger
 */
void cublas_dger_smart_wrapper(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    /* Get device buffer for input x */
    fb_gpu_ptr_t d_x;
    size_t size_x = calculate_vector_size(m, incx) * sizeof(double);
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        return;
    }

    /* Get device buffer for input y */
    fb_gpu_ptr_t d_y;
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        return;
    }

    /* Get device buffer for output a */
    fb_gpu_ptr_t d_a;
    size_t size_a = calculate_matrix_size_col_major(m, n, lda) * sizeof(double);
    if (get_device_buffer_output(a, size_a, &d_a) != 0) {
        return;
    }

    /* Call GPU trait dger */
    /* TODO: Implement trait call for dger */

    /* Mark a as dirty */
    mark_output_dirty(a, d_a);

    /* Release x */
    release_device_buffer(d_x);

    /* Release y */
    release_device_buffer(d_y);

    /* Release a */
    release_device_buffer(d_a);

}


/**
 * @brief Smart wrapper for sgemm
 */
void cublas_sgemm_smart_wrapper(char transa, char transb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) {
    /* Get device buffer for input a */
    fb_gpu_ptr_t d_a;
    size_t size_a = calculate_matrix_size_col_major(transa == 'N' ? m : k, transa == 'N' ? k : m, lda) * sizeof(float);
    if (get_device_buffer_input(a, size_a, &d_a) != 0) {
        return;
    }

    /* Get device buffer for input b */
    fb_gpu_ptr_t d_b;
    size_t size_b = calculate_matrix_size_col_major(transb == 'N' ? k : n, transb == 'N' ? n : k, ldb) * sizeof(float);
    if (get_device_buffer_input(b, size_b, &d_b) != 0) {
        return;
    }

    /* Get device buffer for output c */
    fb_gpu_ptr_t d_c;
    size_t size_c = calculate_matrix_size_col_major(m, n, ldc) * sizeof(float);
    if (get_device_buffer_output(c, size_c, &d_c) != 0) {
        return;
    }

    /* Call GPU trait sgemm */
    /* TODO: Implement trait call for sgemm */

    /* Mark c as dirty */
    mark_output_dirty(c, d_c);

    /* Release a */
    release_device_buffer(d_a);

    /* Release b */
    release_device_buffer(d_b);

    /* Release c */
    release_device_buffer(d_c);

}


/**
 * @brief Smart wrapper for dgemm
 */
void cublas_dgemm_smart_wrapper(char transa, char transb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
    /* Get device buffer for input a */
    fb_gpu_ptr_t d_a;
    size_t size_a = calculate_matrix_size_col_major(transa == 'N' ? m : k, transa == 'N' ? k : m, lda) * sizeof(double);
    if (get_device_buffer_input(a, size_a, &d_a) != 0) {
        return;
    }

    /* Get device buffer for input b */
    fb_gpu_ptr_t d_b;
    size_t size_b = calculate_matrix_size_col_major(transb == 'N' ? k : n, transb == 'N' ? n : k, ldb) * sizeof(double);
    if (get_device_buffer_input(b, size_b, &d_b) != 0) {
        return;
    }

    /* Get device buffer for output c */
    fb_gpu_ptr_t d_c;
    size_t size_c = calculate_matrix_size_col_major(m, n, ldc) * sizeof(double);
    if (get_device_buffer_output(c, size_c, &d_c) != 0) {
        return;
    }

    /* Call GPU trait dgemm */
    /* TODO: Implement trait call for dgemm */

    /* Mark c as dirty */
    mark_output_dirty(c, d_c);

    /* Release a */
    release_device_buffer(d_a);

    /* Release b */
    release_device_buffer(d_b);

    /* Release c */
    release_device_buffer(d_c);

}

