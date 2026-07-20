typedef int CUdevice;
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGet(CUdevice *device, int ordinal);
extern CUresult cuCtxCreate_v2(CUcontext *context, unsigned int flags, CUdevice device);
extern CUresult cuCtxDestroy_v2(CUcontext context);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuModuleLoad(CUmodule *module, const char *path);
extern CUresult cuModuleUnload(CUmodule module);
extern CUresult cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name);
extern CUresult cuMemAlloc_v2(CUdeviceptr *device_ptr, unsigned long bytes);
extern CUresult cuMemFree_v2(CUdeviceptr device_ptr);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr destination, const void *source, unsigned long bytes);
extern CUresult cuMemcpyDtoH_v2(void *destination, CUdeviceptr source, unsigned long bytes);
extern CUresult cuLaunchKernel(CUfunction function,
                               unsigned int grid_x,
                               unsigned int grid_y,
                               unsigned int grid_z,
                               unsigned int block_x,
                               unsigned int block_y,
                               unsigned int block_z,
                               unsigned int shared_memory_bytes,
                               CUstream stream,
                               void **kernel_parameters,
                               void **extra);

enum {
    ELEMENT_COUNT = 4096,
    THREAD_COUNT = 256,
};

static float input_a[ELEMENT_COUNT];
static float input_b[ELEMENT_COUNT];
static float output[ELEMENT_COUNT];

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

static int check(CUresult result, const char *operation)
{
    if (result == 0)
        return 0;

    print_text(operation);
    print_text(" failed rc=");
    print_uint((unsigned int)result);
    print_text("\n");
    return 1;
}

static int run_probe(void)
{
    static const char cubin_path[] =
        "/mnt/probe_nvme0n1p4/models/.lanxin-build/probes/sm120_mul_probe.cubin";
    const unsigned int count = ELEMENT_COUNT;
    const unsigned long bytes = sizeof(output);
    CUdevice device = 0;
    CUcontext context = (CUcontext)0;
    CUmodule module = (CUmodule)0;
    CUfunction function = (CUfunction)0;
    CUdeviceptr a = 0;
    CUdeviceptr b = 0;
    CUdeviceptr result = 0;
    unsigned int failures = 0;

    for (unsigned int i = 0; i < count; ++i) {
        input_a[i] = (float)i * 0.25f;
        input_b[i] = 2.0f;
        output[i] = -1.0f;
    }

    if (check(cuInit(0), "cuInit") ||
        check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check(cuCtxCreate_v2(&context, 0, device), "cuCtxCreate_v2") ||
        check(cuModuleLoad(&module, cubin_path), "cuModuleLoad") ||
        check(cuModuleGetFunction(&function, module, "sm120_mul_probe"), "cuModuleGetFunction") ||
        check(cuMemAlloc_v2(&a, bytes), "cuMemAlloc_v2(a)") ||
        check(cuMemAlloc_v2(&b, bytes), "cuMemAlloc_v2(b)") ||
        check(cuMemAlloc_v2(&result, bytes), "cuMemAlloc_v2(result)") ||
        check(cuMemcpyHtoD_v2(a, input_a, bytes), "cuMemcpyHtoD_v2(a)") ||
        check(cuMemcpyHtoD_v2(b, input_b, bytes), "cuMemcpyHtoD_v2(b)"))
        return 1;

    void *parameters[] = {&a, &b, &result, (void *)&count};
    if (check(cuLaunchKernel(function,
                             (count + THREAD_COUNT - 1) / THREAD_COUNT, 1, 1,
                             THREAD_COUNT, 1, 1, 0, (CUstream)0, parameters, (void **)0),
              "cuLaunchKernel") ||
        check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        check(cuMemcpyDtoH_v2(output, result, bytes), "cuMemcpyDtoH_v2"))
        return 1;

    for (unsigned int i = 0; i < count; ++i) {
        float expected = input_a[i] * input_b[i];
        float delta = output[i] - expected;
        if (delta < -0.00001f || delta > 0.00001f)
            ++failures;
    }

    print_text("kernel.elements=");
    print_uint(count);
    print_text("\nkernel.failures=");
    print_uint(failures);
    print_text("\nkernel.last_x1000=");
    print_uint((unsigned long long)(output[count - 1] * 1000.0f));
    print_text("\nkernel.verified=");
    print_text(failures == 0 ? "yes\n" : "no\n");

    if (result != 0) {
        print_text("cleanup.cuMemFree(result).begin\n");
        check(cuMemFree_v2(result), "cuMemFree_v2(result)");
        print_text("cleanup.cuMemFree(result).end\n");
    }
    if (b != 0) {
        print_text("cleanup.cuMemFree(b).begin\n");
        check(cuMemFree_v2(b), "cuMemFree_v2(b)");
        print_text("cleanup.cuMemFree(b).end\n");
    }
    if (a != 0) {
        print_text("cleanup.cuMemFree(a).begin\n");
        check(cuMemFree_v2(a), "cuMemFree_v2(a)");
        print_text("cleanup.cuMemFree(a).end\n");
    }
    if (module != (CUmodule)0) {
        print_text("cleanup.cuModuleUnload.begin\n");
        check(cuModuleUnload(module), "cuModuleUnload");
        print_text("cleanup.cuModuleUnload.end\n");
    }
    if (context != (CUcontext)0) {
        print_text("cleanup.cuCtxDestroy_v2.begin\n");
        check(cuCtxDestroy_v2(context), "cuCtxDestroy_v2");
        print_text("cleanup.cuCtxDestroy_v2.end\n");
    }
    return failures == 0 ? 0 : 2;
}

__attribute__((noreturn, force_align_arg_pointer)) void _start(void)
{
    sys_exit(run_probe());
}
