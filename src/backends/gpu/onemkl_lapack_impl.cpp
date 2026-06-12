/**
 * @file onemkl_lapack_impl.cpp
 * @brief Intel oneMKL LAPACK implementation (Intel GPUs)
 * 
 * This implementation uses Intel's oneMKL library with SYCL/DPC++ for LAPACK
 * operations on Intel GPUs (Arc, Flex, Max, Iris Xe).
 * 
 * Zero-cost abstraction: Direct SYCL queue calls, compile-time dispatch.
 * 
 * Build:
 *   icpx -fsycl -I include src/backends/gpu/onemkl_lapack_impl.cpp -lmkl_sycl
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <vector>
#include <cstring>

namespace mkl = oneapi::mkl;

/* ============================================================================
 * Helper: Convert char flags to oneMKL enums
 * ========================================================================== */

static inline mkl::transpose convert_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return mkl::transpose::nontrans;
        case 'T': case 't': return mkl::transpose::trans;
        case 'C': case 'c': return mkl::transpose::conjtrans;
        default: return mkl::transpose::nontrans;
    }
}

static inline mkl::uplo convert_uplo(char uplo) {
    switch (uplo) {
        case 'U': case 'u': return mkl::uplo::upper;
        case 'L': case 'l': return mkl::uplo::lower;
        default: return mkl::uplo::upper;
    }
}

static inline mkl::job convert_job(char jobz) {
    switch (jobz) {
        case 'V': case 'v': return mkl::job::vec;
        case 'N': case 'n': return mkl::job::novec;
        case 'A': case 'a': return mkl::job::allvec;
        case 'S': case 's': return mkl::job::somevec;
        case 'O': case 'o': return mkl::job::overwritevec;
        default: return mkl::job::novec;
    }
}

static inline mkl::jobsvd convert_jobsvd(char job) {
    switch (job) {
        case 'A': case 'a': return mkl::jobsvd::vectors;  // All vectors
        case 'S': case 's': return mkl::jobsvd::somevec;  // Some vectors  
        case 'O': case 'o': return mkl::jobsvd::vectorsina;  // Overwrite vectors in A
        case 'N': case 'n': return mkl::jobsvd::novec;  // No vectors
        default: return mkl::jobsvd::novec;
    }
}

/* ============================================================================
 * LAPACK Operations - LU Factorization
 * ========================================================================== */

extern "C" int onemkl_sgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        // oneMKL uses int64_t for ipiv, allocate temp buffer
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(std::min(m, n), *q);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<float>(*q, m, n, lda);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        // Perform LU factorization (returns event for async)
        auto event = mkl::lapack::getrf(*q, m, n, (float*)a, lda, ipiv64, scratchpad, scratchpad_size);
        event.wait();
        
        // Convert ipiv from int64_t to int32_t
        if (ipiv) {
            auto copy_event = q->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(std::min(m, n)), [=](sycl::id<1> i) {
                    ((int*)ipiv)[i] = static_cast<int>(ipiv64[i]);
                });
            });
            copy_event.wait();
        }
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(std::min(m, n), *q);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(*q, m, n, lda);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrf(*q, m, n, (double*)a, lda, ipiv64, scratchpad, scratchpad_size);
        event.wait();
        
        if (ipiv) {
            auto copy_event = q->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(std::min(m, n)), [=](sycl::id<1> i) {
                    ((int*)ipiv)[i] = static_cast<int>(ipiv64[i]);
                });
            });
            copy_event.wait();
        }
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(std::min(m, n), *q);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<std::complex<float>>(*q, m, n, lda);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrf(*q, m, n, (std::complex<float>*)a, lda, ipiv64, scratchpad, scratchpad_size);
        event.wait();
        
        if (ipiv) {
            auto copy_event = q->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(std::min(m, n)), [=](sycl::id<1> i) {
                    ((int*)ipiv)[i] = static_cast<int>(ipiv64[i]);
                });
            });
            copy_event.wait();
        }
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(std::min(m, n), *q);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<std::complex<double>>(*q, m, n, lda);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrf(*q, m, n, (std::complex<double>*)a, lda, ipiv64, scratchpad, scratchpad_size);
        event.wait();
        
        if (ipiv) {
            auto copy_event = q->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(std::min(m, n)), [=](sycl::id<1> i) {
                    ((int*)ipiv)[i] = static_cast<int>(ipiv64[i]);
                });
            });
            copy_event.wait();
        }
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * LAPACK Operations - LU Solve
 * ========================================================================== */

extern "C" int onemkl_sgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                              fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        // Convert ipiv from int32_t to int64_t
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(n, *q);
        auto convert_event = q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                ipiv64[i] = static_cast<int64_t>(((int*)ipiv)[i]);
            });
        });
        convert_event.wait();
        
        auto trans_op = convert_transpose(trans);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrs_scratchpad_size<float>(*q, trans_op, n, nrhs, lda, ldb);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrs(*q, trans_op, n, nrhs, (float*)a, lda, ipiv64, (float*)b, ldb, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                              fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(n, *q);
        auto convert_event = q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                ipiv64[i] = static_cast<int64_t>(((int*)ipiv)[i]);
            });
        });
        convert_event.wait();
        
        auto trans_op = convert_transpose(trans);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrs_scratchpad_size<double>(*q, trans_op, n, nrhs, lda, ldb);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrs(*q, trans_op, n, nrhs, (double*)a, lda, ipiv64, (double*)b, ldb, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                              fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(n, *q);
        auto convert_event = q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                ipiv64[i] = static_cast<int64_t>(((int*)ipiv)[i]);
            });
        });
        convert_event.wait();
        
        auto trans_op = convert_transpose(trans);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrs_scratchpad_size<std::complex<float>>(*q, trans_op, n, nrhs, lda, ldb);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrs(*q, trans_op, n, nrhs, (std::complex<float>*)a, lda, ipiv64, 
                                        (std::complex<float>*)b, ldb, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                              fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t* ipiv64 = sycl::malloc_device<int64_t>(n, *q);
        auto convert_event = q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                ipiv64[i] = static_cast<int64_t>(((int*)ipiv)[i]);
            });
        });
        convert_event.wait();
        
        auto trans_op = convert_transpose(trans);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::getrs_scratchpad_size<std::complex<double>>(*q, trans_op, n, nrhs, lda, ldb);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::getrs(*q, trans_op, n, nrhs, (std::complex<double>*)a, lda, ipiv64, 
                                        (std::complex<double>*)b, ldb, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        sycl::free(ipiv64, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * LAPACK Operations - Cholesky Factorization
 * ========================================================================== */

extern "C" int onemkl_spotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                              fb_gpu_ptr_t a, int lda) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        // Query workspace size
        int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<float>(*q, uplo_flag, n, lda);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        // Perform Cholesky factorization
        auto event = mkl::lapack::potrf(*q, uplo_flag, n, (float*)a, lda, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                              fb_gpu_ptr_t a, int lda) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(*q, uplo_flag, n, lda);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrf(*q, uplo_flag, n, (double*)a, lda, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                              fb_gpu_ptr_t a, int lda) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<std::complex<float>>(*q, uplo_flag, n, lda);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrf(*q, uplo_flag, n, (std::complex<float>*)a, lda, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                              fb_gpu_ptr_t a, int lda) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<std::complex<double>>(*q, uplo_flag, n, lda);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrf(*q, uplo_flag, n, (std::complex<double>*)a, lda, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * LAPACK Operations - Cholesky Solve
 * ========================================================================== */

extern "C" int onemkl_spotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrs_scratchpad_size<float>(*q, uplo_flag, n, nrhs, lda, ldb);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrs(*q, uplo_flag, n, nrhs, (float*)a, lda, (float*)b, ldb, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dpotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrs_scratchpad_size<double>(*q, uplo_flag, n, nrhs, lda, ldb);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrs(*q, uplo_flag, n, nrhs, (double*)a, lda, (double*)b, ldb, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cpotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrs_scratchpad_size<std::complex<float>>(*q, uplo_flag, n, nrhs, lda, ldb);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrs(*q, uplo_flag, n, nrhs, (std::complex<float>*)a, lda, 
                                        (std::complex<float>*)b, ldb, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zpotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::potrs_scratchpad_size<std::complex<double>>(*q, uplo_flag, n, nrhs, lda, ldb);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::potrs(*q, uplo_flag, n, nrhs, (std::complex<double>*)a, lda, 
                                        (std::complex<double>*)b, ldb, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * LAPACK Operations - QR Factorization
 * ========================================================================== */

extern "C" int onemkl_sgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t scratchpad_size = mkl::lapack::geqrf_scratchpad_size<float>(*q, m, n, lda);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        auto event = mkl::lapack::geqrf(*q, m, n, (float*)a, lda, (float*)tau, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t scratchpad_size = mkl::lapack::geqrf_scratchpad_size<double>(*q, m, n, lda);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::geqrf(*q, m, n, (double*)a, lda, (double*)tau, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t scratchpad_size = mkl::lapack::geqrf_scratchpad_size<std::complex<float>>(*q, m, n, lda);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::geqrf(*q, m, n, (std::complex<float>*)a, lda, 
                                        (std::complex<float>*)tau, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                              fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        int64_t scratchpad_size = mkl::lapack::geqrf_scratchpad_size<std::complex<double>>(*q, m, n, lda);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::geqrf(*q, m, n, (std::complex<double>*)a, lda, 
                                        (std::complex<double>*)tau, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * LAPACK Operations - SVD (Note: oneMKL has gesvd, we use it directly)
 * ========================================================================== */

extern "C" int onemkl_sgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                              int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                              fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobu_flag = convert_jobsvd(jobu);
        auto jobvt_flag = convert_jobsvd(jobvt);
        
        int64_t scratchpad_size = mkl::lapack::gesvd_scratchpad_size<float>(*q, jobu_flag, jobvt_flag, 
                                                                              m, n, lda, ldu, ldvt);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        auto event = mkl::lapack::gesvd(*q, jobu_flag, jobvt_flag, m, n, (float*)a, lda, (float*)s,
                                        (float*)u, ldu, (float*)vt, ldvt, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                              int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                              fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobu_flag = convert_jobsvd(jobu);
        auto jobvt_flag = convert_jobsvd(jobvt);
        
        int64_t scratchpad_size = mkl::lapack::gesvd_scratchpad_size<double>(*q, jobu_flag, jobvt_flag, 
                                                                               m, n, lda, ldu, ldvt);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::gesvd(*q, jobu_flag, jobvt_flag, m, n, (double*)a, lda, (double*)s,
                                        (double*)u, ldu, (double*)vt, ldvt, scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                              int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                              fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobu_flag = convert_jobsvd(jobu);
        auto jobvt_flag = convert_jobsvd(jobvt);
        
        int64_t scratchpad_size = mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(*q, jobu_flag, jobvt_flag, 
                                                                                            m, n, lda, ldu, ldvt);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::gesvd(*q, jobu_flag, jobvt_flag, m, n, (std::complex<float>*)a, lda, (float*)s,
                                        (std::complex<float>*)u, ldu, (std::complex<float>*)vt, ldvt, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                              int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                              fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobu_flag = convert_jobsvd(jobu);
        auto jobvt_flag = convert_jobsvd(jobvt);
        
        int64_t scratchpad_size = mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(*q, jobu_flag, jobvt_flag, 
                                                                                             m, n, lda, ldu, ldvt);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::gesvd(*q, jobu_flag, jobvt_flag, m, n, (std::complex<double>*)a, lda, (double*)s,
                                        (std::complex<double>*)u, ldu, (std::complex<double>*)vt, ldvt, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * LAPACK Operations - Eigenvalues (Symmetric/Hermitian)
 * ========================================================================== */

extern "C" int onemkl_ssyev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                             int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobz_flag = convert_job(jobz);
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::syevd_scratchpad_size<float>(*q, jobz_flag, uplo_flag, n, lda);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, *q);
        
        auto event = mkl::lapack::syevd(*q, jobz_flag, uplo_flag, n, (float*)a, lda, (float*)w, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_dsyev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                             int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobz_flag = convert_job(jobz);
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::syevd_scratchpad_size<double>(*q, jobz_flag, uplo_flag, n, lda);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, *q);
        
        auto event = mkl::lapack::syevd(*q, jobz_flag, uplo_flag, n, (double*)a, lda, (double*)w, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_cheev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                             int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobz_flag = convert_job(jobz);
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::heevd_scratchpad_size<std::complex<float>>(*q, jobz_flag, uplo_flag, n, lda);
        std::complex<float>* scratchpad = sycl::malloc_device<std::complex<float>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::heevd(*q, jobz_flag, uplo_flag, n, (std::complex<float>*)a, lda, (float*)w, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

extern "C" int onemkl_zheev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                             int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    sycl::queue* q = (stream) ? (sycl::queue*)stream : (sycl::queue*)handle;
    
    try {
        auto jobz_flag = convert_job(jobz);
        auto uplo_flag = convert_uplo(uplo);
        
        int64_t scratchpad_size = mkl::lapack::heevd_scratchpad_size<std::complex<double>>(*q, jobz_flag, uplo_flag, n, lda);
        std::complex<double>* scratchpad = sycl::malloc_device<std::complex<double>>(scratchpad_size, *q);
        
        auto event = mkl::lapack::heevd(*q, jobz_flag, uplo_flag, n, (std::complex<double>*)a, lda, (double*)w, 
                                        scratchpad, scratchpad_size);
        event.wait();
        
        sycl::free(scratchpad, *q);
        
        return 0;
    } catch (const sycl::exception& e) {
        return -1;
    }
}

/* ============================================================================
 * oneMKL LAPACK Trait Vtable (LAPACK operations only)
 * Combine with onemkl_trait_impl.cpp for full backend
 * ========================================================================== */

extern "C" const fb_gpu_backend_trait_t fb_onemkl_lapack_trait = {
    .name = "Intel oneMKL LAPACK (Intel GPUs)",
    .type = FB_GPU_BACKEND_ONEMKL,
    
    /* LAPACK operations */
    .sgetrf = onemkl_sgetrf,
    .dgetrf = onemkl_dgetrf,
    .cgetrf = onemkl_cgetrf,
    .zgetrf = onemkl_zgetrf,
    
    .sgetrs = onemkl_sgetrs,
    .dgetrs = onemkl_dgetrs,
    .cgetrs = onemkl_cgetrs,
    .zgetrs = onemkl_zgetrs,
    
    .spotrf = onemkl_spotrf,
    .dpotrf = onemkl_dpotrf,
    .cpotrf = onemkl_cpotrf,
    .zpotrf = onemkl_zpotrf,
    
    .spotrs = onemkl_spotrs,
    .dpotrs = onemkl_dpotrs,
    .cpotrs = onemkl_cpotrs,
    .zpotrs = onemkl_zpotrs,
    
    .sgeqrf = onemkl_sgeqrf,
    .dgeqrf = onemkl_dgeqrf,
    .cgeqrf = onemkl_cgeqrf,
    .zgeqrf = onemkl_zgeqrf,
    
    .sgesvd = onemkl_sgesvd,
    .dgesvd = onemkl_dgesvd,
    .cgesvd = onemkl_cgesvd,
    .zgesvd = onemkl_zgesvd,
    
    .ssyev = onemkl_ssyev,
    .dsyev = onemkl_dsyev,
    .cheev = onemkl_cheev,
    .zheev = onemkl_zheev,
};
