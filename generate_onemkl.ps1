# Generate complete oneMKL BLAS backend
$operations = @(
    # Level 1 BLAS
    @{name="axpy"; variants=@("s","d","c","z"); params="int n, {scalar} alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy"; mkl="axpy(*queue, n, alpha, static_cast<const {type}*>(x), incx, static_cast<{type}*>(y), incy)"},
    @{name="scal"; variants=@("s","d","c","z","cs","zd"); params="int n, {scalar} alpha, fb_gpu_ptr_t x, int incx"; mkl="scal(*queue, n, alpha, static_cast<{type}*>(x), incx)"},
    @{name="copy"; variants=@("s","d","c","z"); params="int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy"; mkl="copy(*queue, n, static_cast<const {type}*>(x), incx, static_cast<{type}*>(y), incy)"},
    @{name="swap"; variants=@("s","d","c","z"); params="int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy"; mkl="swap(*queue, n, static_cast<{type}*>(x), incx, static_cast<{type}*>(y), incy)"}
)

$typeMap = @{
    "s" = "float"; "d" = "double"
    "c" = "std::complex<float>"; "z" = "std::complex<double>"
    "cs" = "std::complex<float>"; "zd" = "std::complex<double>"
}

foreach ($op in $operations) {
    foreach ($v in $op.variants) {
        $fullName = "$v$($op.name)"
        $type = $typeMap[$v]
        $scalar = if ($type -match "complex") { "const void*" } else { $type }
        $params = $op.params -replace "{scalar}", $scalar
        $mklCall = $op.mkl -replace "{type}", $type
        
        Write-Output "static void onemkl_${fullName}_impl(void* handle, fb_gpu_stream_t stream, $params) {"
        Write-Output "    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);"
        Write-Output "    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;"
        Write-Output "    try {"
        Write-Output "        oneapi::mkl::blas::$mklCall;"
        Write-Output "    } catch (const oneapi::mkl::exception& e) {"
        Write-Output "        fprintf(stderr, \"oneMKL $fullName error: %s\n\", e.what());"
        Write-Output "    }"
        Write-Output "}"
        Write-Output ""
    }
}
