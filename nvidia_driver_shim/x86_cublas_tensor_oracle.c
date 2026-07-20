typedef int CUdevice;
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUevent;
typedef void *cublasHandle_t;
typedef int cublasStatus_t;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGet(CUdevice *device, int ordinal);
extern CUresult cuCtxCreate_v2(CUcontext *context, unsigned int flags, CUdevice device);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuMemAlloc_v2(CUdeviceptr *device_ptr, unsigned long bytes);
extern CUresult cuMemsetD16_v2(CUdeviceptr destination, unsigned short value, unsigned long count);
extern CUresult cuMemcpyDtoH_v2(void *destination, CUdeviceptr source, unsigned long bytes);
extern CUresult cuEventCreate(CUevent *event, unsigned int flags);
extern CUresult cuEventRecord(CUevent event, void *stream);
extern CUresult cuEventSynchronize(CUevent event);
extern CUresult cuEventElapsedTime(float *milliseconds, CUevent start, CUevent end);

extern cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
extern cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode);
extern cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                                   int trans_a,
                                   int trans_b,
                                   int m,
                                   int n,
                                   int k,
                                   const void *alpha,
                                   const void *a,
                                   int a_type,
                                   int lda,
                                   const void *b,
                                   int b_type,
                                   int ldb,
                                   const void *beta,
                                   void *c,
                                   int c_type,
                                   int ldc,
                                   int compute_type,
                                   int algorithm);

enum {
    MATRIX_DIMENSION = 16384,
    REPETITIONS = 3,
    CUDA_R_16F = 2,
    CUBLAS_OP_N = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_GEMM_DEFAULT = -1,
};

static long sys_write(int fd, const void *buffer, unsigned long bytes)
{
    register long rax __asm__("rax") = 1;
    register long rdi __asm__("rdi") = fd;
    register long rsi __asm__("rsi") = (long)buffer;
    register long rdx __asm__("rdx") = (long)bytes;
    __asm__ volatile("syscall"
                     : "+a"(rax)
                     : "D"(rdi), "S"(rsi), "d"(rdx)
                     : "rcx", "r11", "memory");
    return rax;
}

static __attribute__((noreturn)) void sys_exit(int status)
{
    register long rax __asm__("rax") = 60;
    register long rdi __asm__("rdi") = status;
    __asm__ volatile("syscall" : "+a"(rax) : "D"(rdi) : "rcx", "r11", "memory");
    __builtin_unreachable();
}

static unsigned long text_len(const char *text)
{
    unsigned long len = 0;
    while (text[len] != '\0')
        ++len;
    return len;
}

static void print_text(const char *text)
{
    sys_write(1, text, text_len(text));
}

static void print_uint(unsigned long long value)
{
    char buffer[24];
    unsigned int digits = 0;

    do {
        buffer[digits++] = (char)('0' + value % 10ULL);
        value /= 10ULL;
    } while (value != 0ULL);

    for (unsigned int left = 0, right = digits - 1; left < right; ++left, --right) {
        char tmp = buffer[left];
        buffer[left] = buffer[right];
        buffer[right] = tmp;
    }
    sys_write(1, buffer, digits);
}

static int check_cuda(CUresult result, const char *operation)
{
    if (result == 0)
        return 0;
    print_text(operation);
    print_text(" failed rc=");
    print_uint((unsigned int)result);
    print_text("\n");
    return 1;
}

static int check_cublas(cublasStatus_t result, const char *operation)
{
    if (result == 0)
        return 0;
    print_text(operation);
    print_text(" failed rc=");
    print_uint((unsigned int)result);
    print_text("\n");
    return 1;
}

static int run_benchmark(void)
{
    const int dimension = MATRIX_DIMENSION;
    const unsigned long elements = (unsigned long)dimension * (unsigned long)dimension;
    const unsigned long bytes = elements * sizeof(unsigned short);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUdevice device = 0;
    CUcontext context = (CUcontext)0;
    CUdeviceptr a = 0;
    CUdeviceptr b = 0;
    CUdeviceptr c = 0;
    CUevent start = (CUevent)0;
    CUevent stop = (CUevent)0;
    cublasHandle_t handle = (cublasHandle_t)0;
    float elapsed_ms = 0.0f;
    unsigned short sample = 0;

    if (check_cuda(cuInit(0), "cuInit") ||
        check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check_cuda(cuCtxCreate_v2(&context, 0, device), "cuCtxCreate_v2") ||
        check_cublas(cublasCreate_v2(&handle), "cublasCreate_v2") ||
        check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode") ||
        check_cuda(cuMemAlloc_v2(&a, bytes), "cuMemAlloc_v2(a)") ||
        check_cuda(cuMemAlloc_v2(&b, bytes), "cuMemAlloc_v2(b)") ||
        check_cuda(cuMemAlloc_v2(&c, bytes), "cuMemAlloc_v2(c)") ||
        check_cuda(cuMemsetD16_v2(a, 0x3c00, elements), "cuMemsetD16_v2(a)") ||
        check_cuda(cuMemsetD16_v2(b, 0x3c00, elements), "cuMemsetD16_v2(b)") ||
        check_cuda(cuMemsetD16_v2(c, 0, elements), "cuMemsetD16_v2(c)"))
        return 1;

    if (check_cublas(cublasGemmEx(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  dimension, dimension, dimension,
                                  &alpha,
                                  (const void *)(unsigned long)a, CUDA_R_16F, dimension,
                                  (const void *)(unsigned long)b, CUDA_R_16F, dimension,
                                  &beta,
                                  (void *)(unsigned long)c, CUDA_R_16F, dimension,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx(warmup)") ||
        check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(warmup)") ||
        check_cuda(cuEventCreate(&start, 0), "cuEventCreate(start)") ||
        check_cuda(cuEventCreate(&stop, 0), "cuEventCreate(stop)") ||
        check_cuda(cuEventRecord(start, (void *)0), "cuEventRecord(start)"))
        return 1;

    for (int repetition = 0; repetition < REPETITIONS; ++repetition) {
        if (check_cublas(cublasGemmEx(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      dimension, dimension, dimension,
                                      &alpha,
                                      (const void *)(unsigned long)a, CUDA_R_16F, dimension,
                                      (const void *)(unsigned long)b, CUDA_R_16F, dimension,
                                      &beta,
                                      (void *)(unsigned long)c, CUDA_R_16F, dimension,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                         "cublasGemmEx"))
            return 1;
    }

    if (check_cuda(cuEventRecord(stop, (void *)0), "cuEventRecord(stop)") ||
        check_cuda(cuEventSynchronize(stop), "cuEventSynchronize(stop)") ||
        check_cuda(cuEventElapsedTime(&elapsed_ms, start, stop), "cuEventElapsedTime") ||
        check_cuda(cuMemcpyDtoH_v2(&sample, c, sizeof(sample)), "cuMemcpyDtoH_v2(sample)"))
        return 1;

    const unsigned long long operations_per_gemm =
        2ULL * (unsigned long long)dimension * (unsigned long long)dimension *
        (unsigned long long)dimension;
    const double tflops = ((double)operations_per_gemm * (double)REPETITIONS) /
                          ((double)elapsed_ms * 1000000000.0);

    print_text("gemm.dimension=");
    print_uint(dimension);
    print_text("\ngemm.repetitions=");
    print_uint(REPETITIONS);
    print_text("\ngemm.elapsed_us=");
    print_uint((unsigned long long)(elapsed_ms * 1000.0f));
    print_text("\ngemm.tflops_x100=");
    print_uint((unsigned long long)(tflops * 100.0));
    print_text("\ngemm.sample_half_bits=");
    print_uint(sample);
    print_text("\ngemm.verified=");
    print_text(sample == 0x7400 ? "yes\n" : "no\n");
    return sample == 0x7400 ? 0 : 2;
}

__attribute__((noreturn, force_align_arg_pointer)) void _start(void)
{
    // Process teardown is intentionally delegated to the kernel. The current
    // RISC-V port can spend minutes in explicit CUDA context destruction.
    sys_exit(run_benchmark());
}
